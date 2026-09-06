#pragma once

#include "neural_net.hpp"

#include <memory>
#include <string>

// Batched inference
// evaluate() is thread-safe, many self-play game
// threads call it concurrently and each call enqueues one leaf position,
// blocks, and a single background thread runs batched forward passes on the GPU
// and wakes the callers with their results
class NeuralNetwork {
public:
  explicit NeuralNetwork(const std::string &model_path,
                         int max_batch_size = 256, int batch_timeout_us = 1000);
  ~NeuralNetwork();
  NeuralNetwork(const NeuralNetwork &) = delete;
  NeuralNetwork &operator=(const NeuralNetwork &) = delete;

  // Blocks until this position is evaluated as part of a batch
  NetworkOutput evaluate(const Position &position);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};