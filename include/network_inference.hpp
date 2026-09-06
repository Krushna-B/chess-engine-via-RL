#pragma once

#include "neural_net.hpp"

#include <array>
#include <memory>
#include <string>

// Model-usage / throughput / latency telemetry, accumulated by the batching
// server over its lifetime. Read once at the end of a generation.
struct InferenceStats {
  std::string device;
  long long eval_count = 0;   // total leaf evaluations
  long long batch_count = 0;  // total forward passes
  long long max_batch = 0;    // largest batch seen
  double mean_batch = 0.0;    // eval_count / batch_count
  double mean_forward_ms = 0.0; // mean per-batch forward-pass latency
  double max_forward_ms = 0.0;
  double mean_wait_ms = 0.0;  // mean time a request waited in the queue
  double max_wait_ms = 0.0;
  // Batch-size histogram, bucketed by power of two: [1],[2-3],[4-7],... clamped.
  std::array<long long, 9> batch_buckets{};
};

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

  // Snapshot of accumulated telemetry (call when self-play is done).
  InferenceStats stats();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};