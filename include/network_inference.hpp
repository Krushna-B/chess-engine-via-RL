#pragma once

#include "neural_net.hpp"

#include <memory>
#include <string>

class NeuralNetwork {
public:
  explicit NeuralNetwork(const std::string &model_path);
  ~NeuralNetwork();
  NeuralNetwork(const NeuralNetwork &) = delete;
  NeuralNetwork &operator=(const NeuralNetwork &) = delete;
  NetworkOutput evaluate(const Position &position);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};