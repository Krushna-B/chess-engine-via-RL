#include "training_data.hpp"

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

constexpr std::uint32_t DATASET_MAGIC = 0x43485A31;
constexpr std::uint32_t DATASET_VERSION = 1;

template <typename T> void write_value(std::ofstream &output, const T &value) {
  output.write(reinterpret_cast<const char *>(&value),
               static_cast<std::streamsize>(sizeof(T)));

  if (!output) {
    throw std::runtime_error("Failed to write training data");
  }
}

template <typename T> void read_value(std::ifstream &input, T &value) {
  input.read(reinterpret_cast<char *>(&value),
             static_cast<std::streamsize>(sizeof(T)));

  if (!input) {
    throw std::runtime_error("Failed to read training data");
  }
}

void write_float_array(std::ofstream &output, const float *data,
                       std::size_t count) {
  output.write(reinterpret_cast<const char *>(data),
               static_cast<std::streamsize>(count * sizeof(float)));

  if (!output) {
    throw std::runtime_error("Failed to write float array");
  }
}

void read_float_array(std::ifstream &input, float *data, std::size_t count) {
  input.read(reinterpret_cast<char *>(data),
             static_cast<std::streamsize>(count * sizeof(float)));

  if (!input) {
    throw std::runtime_error("Failed to read float array");
  }
}

void save_training_examples(const std::string &filename,
                            const std::vector<TrainingExample> &examples) {
  static_assert(sizeof(float) == 4);

  std::ofstream output(filename, std::ios::binary | std::ios::trunc);

  if (!output.is_open()) {
    throw std::runtime_error("Could not open output file: " + filename);
  }

  const std::uint32_t state_size =
      static_cast<std::uint32_t>(ENCODED_STATE_SIZE);

  const std::uint32_t policy_size = static_cast<std::uint32_t>(POLICY_SIZE);

  const std::uint64_t example_count =
      static_cast<std::uint64_t>(examples.size());

  // File header
  write_value(output, DATASET_MAGIC);
  write_value(output, DATASET_VERSION);
  write_value(output, state_size);
  write_value(output, policy_size);
  write_value(output, example_count);

  // Training records
  for (const TrainingExample &example : examples) {
    write_float_array(output, example.encoded_position.data(),
                      example.encoded_position.size());

    write_float_array(output, example.policy_target.data(),
                      example.policy_target.size());

    write_value(output, example.value_target);
  }
}
std::vector<TrainingExample>
load_training_examples(const std::string &filename) {
  static_assert(sizeof(float) == 4);

  std::ifstream input(filename, std::ios::binary);

  if (!input.is_open()) {
    throw std::runtime_error("Could not open input file: " + filename);
  }

  std::uint32_t magic{};
  std::uint32_t version{};
  std::uint32_t state_size{};
  std::uint32_t policy_size{};
  std::uint64_t example_count{};

  read_value(input, magic);
  read_value(input, version);
  read_value(input, state_size);
  read_value(input, policy_size);
  read_value(input, example_count);

  if (magic != DATASET_MAGIC) {
    throw std::runtime_error("Invalid training-data file");
  }

  if (version != DATASET_VERSION) {
    throw std::runtime_error("Unsupported dataset version");
  }

  if (state_size != ENCODED_STATE_SIZE) {
    throw std::runtime_error("Encoded state size does not match");
  }

  if (policy_size != POLICY_SIZE) {
    throw std::runtime_error("Policy size does not match");
  }

  std::vector<TrainingExample> examples;
  examples.resize(static_cast<std::size_t>(example_count));

  for (TrainingExample &example : examples) {
    read_float_array(input, example.encoded_position.data(),
                     example.encoded_position.size());

    read_float_array(input, example.policy_target.data(),
                     example.policy_target.size());

    read_value(input, example.value_target);
  }

  return examples;
}
