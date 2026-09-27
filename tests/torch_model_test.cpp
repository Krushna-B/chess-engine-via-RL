#include <torch/script.h>
#include <torch/torch.h>

#include <exception>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: torch_model_test <model-path>\n";

    return 1;
  }

  try {
    const std::string model_path = argv[1];

    torch::jit::script::Module model =
        torch::jit::load(model_path, torch::kCPU);

    model.eval();

    c10::InferenceMode inference_mode;

    torch::Tensor input = torch::zeros(
        {1, 64, 18},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

    std::vector<torch::jit::IValue> inputs;
    inputs.emplace_back(input);

    torch::jit::IValue output = model.forward(inputs);

    if (!output.isTuple()) {
      throw std::runtime_error("Model did not return a tuple");
    }

    auto output_tuple = output.toTuple();
    const auto &elements = output_tuple->elements();

    if (elements.size() != 2) {
      throw std::runtime_error("Expected policy and value outputs");
    }

    torch::Tensor policy_logits = elements[0].toTensor().contiguous();

    torch::Tensor value = elements[1].toTensor().contiguous();

    std::cout << "Policy shape: " << policy_logits.sizes() << '\n';

    std::cout << "Value shape: " << value.sizes() << '\n';

    std::cout << "Value: " << value.item<float>() << '\n';

    if (policy_logits.numel() != 4672) {
      throw std::runtime_error("Expected 4672 policy logits");
    }

    if (value.numel() != 1) {
      throw std::runtime_error("Expected one value output");
    }

    std::cout << "C++ LibTorch inference passed\n";

  } catch (const c10::Error &error) {
    std::cerr << "LibTorch error:\n" << error.what() << '\n';

    return 1;

  } catch (const std::exception &error) {
    std::cerr << "Error: " << error.what() << '\n';

    return 1;
  }

  return 0;
}