#include "network_inference.hpp"

#include <torch/script.h>
#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace {

struct PendingRequest {
  EncodedPosition input;
  std::promise<NetworkOutput> result;
};

} // namespace

struct NeuralNetwork::Impl {
  torch::jit::script::Module model;
  torch::Device device;
  std::size_t max_batch_size;
  std::chrono::microseconds batch_timeout;

  std::mutex mutex;
  std::condition_variable cv;
  std::queue<PendingRequest> pending;
  bool shutdown = false;
  std::thread worker;

  Impl(const std::string &model_path, int max_batch, int timeout_us)
      : model(torch::jit::load(model_path, torch::kCPU)),
        device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
        max_batch_size(static_cast<std::size_t>(max_batch)),
        batch_timeout(timeout_us) {
    model.to(device);
    model.eval();
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

  std::future<NetworkOutput> submit(const Position &position) {
    PendingRequest request;
    request.input = encode_position(position);
    std::future<NetworkOutput> future = request.result.get_future();

    {
      std::lock_guard<std::mutex> lock(mutex);
      pending.push(std::move(request));
    }
    cv.notify_one();

    return future;
  }

  // Background thread: collect a batch, run one forward pass, fulfil promises
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

    try {
      std::vector<float> buffer(batch_size * ENCODED_STATE_SIZE);
      for (std::size_t i = 0; i < batch_size; ++i) {
        std::copy(batch[i].input.begin(), batch[i].input.end(),
                  buffer.begin() +
                      static_cast<std::ptrdiff_t>(i * ENCODED_STATE_SIZE));
      }

      torch::Tensor input =
          torch::from_blob(buffer.data(),
                           {static_cast<long>(batch_size), 64, 18},
                           torch::TensorOptions().dtype(torch::kFloat32))
              .to(device);

      c10::InferenceMode inference_mode;

      std::vector<torch::jit::IValue> inputs;
      inputs.emplace_back(input);

      const auto tuple = model.forward(inputs).toTuple();
      const auto &elements = tuple->elements();
      if (elements.size() != 2) {
        throw std::runtime_error("Expected policy and value outputs");
      }

      torch::Tensor logits = elements[0]
                                 .toTensor()
                                 .to(torch::kCPU)
                                 .to(torch::kFloat32)
                                 .contiguous();
      torch::Tensor values = elements[1]
                                 .toTensor()
                                 .to(torch::kCPU)
                                 .to(torch::kFloat32)
                                 .contiguous();

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
