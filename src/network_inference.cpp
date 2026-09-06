#include "network_inference.hpp"

#include <torch/script.h>
#include <torch/torch.h>

// CUDA graph support only exists in a real CUDA toolchain. The torch wheel
// ships the ATen/cuda headers even on CPU-only platforms, so gate on the actual
// CUDA SDK header (cuda_runtime_api.h) being present -- absent on macOS/CPU.
#if defined(__has_include)
#if __has_include(<cuda_runtime_api.h>) && __has_include(<ATen/cuda/CUDAGraph.h>)
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#define HAVE_CUDA_GRAPH 1
#endif
#endif

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdlib>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct PendingRequest {
  EncodedPosition input;
  std::promise<NetworkOutput> result;
  Clock::time_point enqueue_time;
};

double ms_between(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

std::size_t batch_bucket(std::size_t batch_size) {
  std::size_t bucket = 0;
  while (batch_size > 1 && bucket < 8) {
    batch_size >>= 1;
    ++bucket;
  }
  return bucket;
}

bool env_disabled(const char *name) {
  const char *value = std::getenv(name);
  return value != nullptr && std::string(value) == "0";
}

} // namespace

struct NeuralNetwork::Impl {
  torch::jit::script::Module model;
  torch::Device device;
  bool use_bf16 = false;
  std::size_t max_batch_size;
  std::chrono::microseconds batch_timeout;

  // CUDA-graph inference: capture the fixed-shape forward once and replay it,
  // collapsing ~dozens of kernel launches per forward into one graph launch.
  bool use_graph = false;
  torch::Tensor graph_input;
  torch::Tensor graph_logits;
  torch::Tensor graph_values;
#ifdef HAVE_CUDA_GRAPH
  std::unique_ptr<at::cuda::CUDAGraph> cuda_graph;
#endif

  std::mutex mutex;
  std::condition_variable cv;
  std::queue<PendingRequest> pending;
  bool shutdown = false;
  std::thread worker;

  // Telemetry. Written only by the worker thread; guarded so stats() can read.
  std::mutex stats_mutex;
  long long eval_count = 0;
  long long batch_count = 0;
  long long max_batch = 0;
  double total_forward_ms = 0.0;
  double max_forward_ms = 0.0;
  double total_wait_ms = 0.0;
  double max_wait_ms = 0.0;
  std::array<long long, 9> batch_buckets{};

  Impl(const std::string &model_path, int max_batch, int timeout_us)
      : model(torch::jit::load(model_path, torch::kCPU)),
        device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
        max_batch_size(static_cast<std::size_t>(max_batch)),
        batch_timeout(timeout_us) {
    model.to(device);
    // bf16 on GPU: big throughput win for a small model that is otherwise
    // kernel-launch / overhead bound. CPU stays float32.
    use_bf16 = device.is_cuda();
    if (use_bf16) {
      model.to(torch::kBFloat16);
    }
    model.eval();

#ifdef HAVE_CUDA_GRAPH
    if (device.is_cuda() && !env_disabled("SELFPLAY_CUDA_GRAPH")) {
      try {
        setup_cuda_graph();
        use_graph = true;
        std::cerr << "[inference] CUDA graph enabled (batch=" << max_batch_size
                  << ")\n";
      } catch (const std::exception &error) {
        use_graph = false;
        std::cerr << "[inference] CUDA graph capture failed, using eager path: "
                  << error.what() << "\n";
      }
    }
#endif

    worker = std::thread([this] { run(); });
  }

  ~Impl() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      shutdown = true;
    }
    cv.notify_all();
    if (worker.joinable()) {
      worker.join();
    }
  }

#ifdef HAVE_CUDA_GRAPH
  // Capture the forward at the fixed max batch size. Smaller batches pad up to
  // this size and ignore the extra rows.
  void setup_cuda_graph() {
    const auto dtype = use_bf16 ? torch::kBFloat16 : torch::kFloat32;
    graph_input =
        torch::zeros({static_cast<long>(max_batch_size), 64, 18},
                     torch::TensorOptions().device(device).dtype(dtype));

    torch::NoGradGuard no_grad;
    std::vector<torch::jit::IValue> inputs{graph_input};

    // Warmup + capture must run on a non-default stream.
    at::cuda::CUDAStream stream = at::cuda::getStreamFromPool();
    c10::cuda::CUDAStreamGuard guard(stream);

    for (int i = 0; i < 3; ++i) {
      model.forward(inputs);
    }
    stream.synchronize();

    cuda_graph = std::make_unique<at::cuda::CUDAGraph>();
    cuda_graph->capture_begin();
    const auto out = model.forward(inputs).toTuple();
    graph_logits = out->elements()[0].toTensor();
    graph_values = out->elements()[1].toTensor();
    cuda_graph->capture_end();
    stream.synchronize();
  }
#endif

  std::future<NetworkOutput> submit(const Position &position) {
    PendingRequest request;
    request.input = encode_position(position);
    request.enqueue_time = Clock::now();
    std::future<NetworkOutput> future = request.result.get_future();

    {
      std::lock_guard<std::mutex> lock(mutex);
      pending.push(std::move(request));
    }
    cv.notify_one();

    return future;
  }

  InferenceStats snapshot() {
    std::lock_guard<std::mutex> lock(stats_mutex);
    InferenceStats out;
    out.device = device.is_cuda()
                     ? (use_graph ? "cuda-bf16-graph"
                                  : (use_bf16 ? "cuda-bf16" : "cuda"))
                     : "cpu";
    out.eval_count = eval_count;
    out.batch_count = batch_count;
    out.max_batch = max_batch;
    out.mean_batch =
        batch_count > 0 ? static_cast<double>(eval_count) / batch_count : 0.0;
    out.mean_forward_ms =
        batch_count > 0 ? total_forward_ms / batch_count : 0.0;
    out.max_forward_ms = max_forward_ms;
    out.mean_wait_ms = eval_count > 0 ? total_wait_ms / eval_count : 0.0;
    out.max_wait_ms = max_wait_ms;
    out.batch_buckets = batch_buckets;
    return out;
  }

  // Background thread: collect a batch, run one forward pass, fulfil promises.
  void run() {
    while (true) {
      std::vector<PendingRequest> batch;

      {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [this] { return shutdown || !pending.empty(); });

        if (shutdown && pending.empty()) {
          return;
        }

        // A batch has started forming, wait briefly for it to grow, but don't
        // stall the GPU waiting for a full batch that may never arrive
        if (pending.size() < max_batch_size) {
          cv.wait_for(lock, batch_timeout, [this] {
            return shutdown || pending.size() >= max_batch_size;
          });
        }

        while (!pending.empty() && batch.size() < max_batch_size) {
          batch.push_back(std::move(pending.front()));
          pending.pop();
        }
      }

      if (!batch.empty()) {
        run_batch(batch);
      }
    }
  }

  void run_batch(std::vector<PendingRequest> &batch) {
    const std::size_t batch_size = batch.size();
    const Clock::time_point batch_start = Clock::now();

    try {
      std::vector<float> buffer(batch_size * ENCODED_STATE_SIZE);
      for (std::size_t i = 0; i < batch_size; ++i) {
        std::copy(batch[i].input.begin(), batch[i].input.end(),
                  buffer.begin() +
                      static_cast<std::ptrdiff_t>(i * ENCODED_STATE_SIZE));
      }

      torch::Tensor host = torch::from_blob(
          buffer.data(), {static_cast<long>(batch_size), 64, 18},
          torch::TensorOptions().dtype(torch::kFloat32));

      torch::Tensor logits;
      torch::Tensor values;

#ifdef HAVE_CUDA_GRAPH
      if (use_graph) {
        const auto dtype = use_bf16 ? torch::kBFloat16 : torch::kFloat32;
        // Copy inputs into the captured static buffer in place, then replay.
        graph_input.slice(0, 0, static_cast<long>(batch_size))
            .copy_(host.to(device).to(dtype));
        cuda_graph->replay();
        at::cuda::getCurrentCUDAStream().synchronize();

        logits = graph_logits.slice(0, 0, static_cast<long>(batch_size))
                     .to(torch::kCPU)
                     .to(torch::kFloat32)
                     .contiguous();
        values = graph_values.slice(0, 0, static_cast<long>(batch_size))
                     .to(torch::kCPU)
                     .to(torch::kFloat32)
                     .contiguous();
      } else
#endif
      {
        torch::Tensor input = host.to(device);
        if (use_bf16) {
          input = input.to(torch::kBFloat16);
        }

        c10::InferenceMode inference_mode;
        std::vector<torch::jit::IValue> inputs;
        inputs.emplace_back(input);

        const auto tuple = model.forward(inputs).toTuple();
        const auto &elements = tuple->elements();
        if (elements.size() != 2) {
          throw std::runtime_error("Expected policy and value outputs");
        }

        logits = elements[0]
                     .toTensor()
                     .to(torch::kCPU)
                     .to(torch::kFloat32)
                     .contiguous();
        values = elements[1]
                     .toTensor()
                     .to(torch::kCPU)
                     .to(torch::kFloat32)
                     .contiguous();
      }

      const float *logits_ptr = logits.data_ptr<float>();
      const float *values_ptr = values.data_ptr<float>();

      for (std::size_t i = 0; i < batch_size; ++i) {
        NetworkOutput output{};
        std::copy_n(logits_ptr + i * POLICY_SIZE,
                    static_cast<std::size_t>(POLICY_SIZE),
                    output.policy_logits.begin());
        output.value = values_ptr[i];
        batch[i].result.set_value(std::move(output));
      }
    } catch (...) {
      for (auto &request : batch) {
        request.result.set_exception(std::current_exception());
      }
    }

    const Clock::time_point batch_end = Clock::now();
    record_batch(batch, batch_start, batch_end);
  }

  void record_batch(const std::vector<PendingRequest> &batch,
                    Clock::time_point start, Clock::time_point end) {
    const std::size_t batch_size = batch.size();
    const double forward_ms = ms_between(start, end);

    std::lock_guard<std::mutex> lock(stats_mutex);
    eval_count += static_cast<long long>(batch_size);
    ++batch_count;
    max_batch = std::max(max_batch, static_cast<long long>(batch_size));
    total_forward_ms += forward_ms;
    max_forward_ms = std::max(max_forward_ms, forward_ms);
    ++batch_buckets[batch_bucket(batch_size)];

    for (const PendingRequest &request : batch) {
      const double wait_ms = ms_between(request.enqueue_time, start);
      total_wait_ms += wait_ms;
      max_wait_ms = std::max(max_wait_ms, wait_ms);
    }
  }
};

NeuralNetwork::NeuralNetwork(const std::string &model_path, int max_batch_size,
                             int batch_timeout_us)
    : impl_(std::make_unique<Impl>(model_path, max_batch_size,
                                   batch_timeout_us)) {}

NeuralNetwork::~NeuralNetwork() = default;

NetworkOutput NeuralNetwork::evaluate(const Position &position) {
  return impl_->submit(position).get();
}

InferenceStats NeuralNetwork::stats() { return impl_->snapshot(); }
