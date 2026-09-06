#include "network_inference.hpp"

#include <torch/script.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

struct NeuralNetwork::Impl {
  torch::jit::script::Module model;

  explicit Impl(const std::string &model_path)
      : model(torch::jit::load(model_path, torch::kCPU)) {
    model.eval();
  }
};

NeuralNetwork::NeuralNetwork(const std::string &model_path)
    : impl_(std::make_unique<Impl>(model_path)) {}

NeuralNetwork::~NeuralNetwork() = default;

NetworkOutput NeuralNetwork::evaluate(const Position &position) {
  EncodedPosition encoded = encode_position(position);

  // EncodedPosition is flat in memory but in tensor it is [64, 18]
  torch::Tensor input =
      torch::from_blob(
          encoded.data(), {1, 64, 18},
          torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
          .clone();

  c10::InferenceMode inference_mode;

  std::vector<torch::jit::IValue> inputs;
  inputs.emplace_back(input);

  torch::jit::IValue result = impl_->model.forward(inputs);

  if (!result.isTuple()) {
    throw std::runtime_error("Neural network did not return a tuple");
  }

  const auto tuple = result.toTuple();
  const auto &elements = tuple->elements();

  if (elements.size() != 2) {
    throw std::runtime_error("Expected policy and value outputs");
  }

  torch::Tensor logits = elements[0]
                             .toTensor()
                             .to(torch::kCPU)
                             .to(torch::kFloat32)
                             .contiguous()
                             .view({-1});

  torch::Tensor value = elements[1]
                            .toTensor()
                            .to(torch::kCPU)
                            .to(torch::kFloat32)
                            .contiguous()
                            .view({-1});

  if (logits.numel() != POLICY_SIZE) {
    throw std::runtime_error("Expected 4672 policy logits");
  }

  if (value.numel() != 1) {
    throw std::runtime_error("Expected one value");
  }

  NetworkOutput output{};

  std::copy_n(logits.data_ptr<float>(), static_cast<std::size_t>(POLICY_SIZE),
              output.policy_logits.begin());

  output.value = value.item<float>();

  if (!std::isfinite(output.value)) {
    throw std::runtime_error("Model returned a non-finite value");
  }

  for (float logit : output.policy_logits) {
    if (!std::isfinite(logit)) {
      throw std::runtime_error("Model returned a non-finite policy logit");
    }
  }

  return output;
}