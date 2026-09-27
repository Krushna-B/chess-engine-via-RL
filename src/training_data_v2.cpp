#include "training_data_v2.hpp"

#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace {

template <typename T> void write_value(std::ofstream &output, const T &value) {
  output.write(reinterpret_cast<const char *>(&value),
               static_cast<std::streamsize>(sizeof(T)));

  if (!output) {
    throw std::runtime_error("Failed to write compact training data");
  }
}

template <typename T> void read_value(std::ifstream &input, T &value) {
  input.read(reinterpret_cast<char *>(&value),
             static_cast<std::streamsize>(sizeof(T)));

  if (!input) {
    throw std::runtime_error("Failed to read compact training data");
  }
}

void validate_example(const CompactTrainingExample &example) {
  if (example.castling_rights > 0x0F) {
    throw std::runtime_error("Invalid castling rights in compact record");
  }

  if (example.side_to_move > 1) {
    throw std::runtime_error("Invalid side to move in compact record");
  }

  if (example.policy.size() > COMPACT_POLICY_SIZE) {
    throw std::runtime_error("Too many policy entries in compact record");
  }

  std::array<bool, COMPACT_POLICY_SIZE> used{};

  for (const CompactPolicyEntry &entry : example.policy) {
    if (entry.index >= COMPACT_POLICY_SIZE || used[entry.index]) {
      throw std::runtime_error("Invalid policy index in compact record");
    }

    if (!std::isfinite(entry.probability) || entry.probability <= 0.0f) {
      throw std::runtime_error("Invalid policy probability in compact record");
    }

    used[entry.index] = true;
  }

  for (float value : example.wdl) {
    if (!std::isfinite(value) || value < 0.0f) {
      throw std::runtime_error("Invalid WDL target in compact record");
    }
  }

  if (!std::isfinite(example.moves_left) || example.moves_left < 0.0f) {
    throw std::runtime_error("Invalid moves-left target in compact record");
  }

  if (example.source != TrainingSource::LC0 &&
      example.source != TrainingSource::SELF_PLAY) {
    throw std::runtime_error("Invalid source in compact record");
  }
}

} // namespace

void save_compact_training_examples(
    const std::string &filename,
    const std::vector<CompactTrainingExample> &examples) {
  static_assert(sizeof(float) == 4);
  static_assert(sizeof(std::uint64_t) == 8);

  std::ofstream output(filename, std::ios::binary | std::ios::trunc);

  if (!output.is_open()) {
    throw std::runtime_error("Could not open output file: " + filename);
  }

  for (const CompactTrainingExample &example : examples) {
    validate_example(example);
  }

  std::uint32_t input_planes = COMPACT_INPUT_PLANES;
  std::uint32_t bitplanes = COMPACT_BITPLANES;
  std::uint32_t policy_size = COMPACT_POLICY_SIZE;
  std::uint64_t example_count = examples.size();

  write_value(output, COMPACT_DATASET_MAGIC);
  write_value(output, COMPACT_DATASET_VERSION);
  write_value(output, input_planes);
  write_value(output, bitplanes);
  write_value(output, policy_size);
  write_value(output, example_count);

  for (const CompactTrainingExample &example : examples) {
    write_value(output, example.game_id);
    write_value(output, example.ply);

    std::uint8_t source = static_cast<std::uint8_t>(example.source);
    std::uint8_t reserved = 0;
    std::uint16_t policy_count =
        static_cast<std::uint16_t>(example.policy.size());

    write_value(output, source);
    write_value(output, example.castling_rights);
    write_value(output, example.side_to_move);
    write_value(output, reserved);
    write_value(output, example.rule50_count);
    write_value(output, policy_count);

    for (std::uint64_t plane : example.planes) {
      write_value(output, plane);
    }
    for (float value : example.wdl) {
      write_value(output, value);
    }
    write_value(output, example.moves_left);

    for (const CompactPolicyEntry &entry : example.policy) {
      write_value(output, entry.index);
      write_value(output, entry.probability);
    }
  }
}

std::vector<CompactTrainingExample>
load_compact_training_examples(const std::string &filename) {
  static_assert(sizeof(float) == 4);
  static_assert(sizeof(std::uint64_t) == 8);

  std::ifstream input(filename, std::ios::binary);

  if (!input.is_open()) {
    throw std::runtime_error("Could not open input file: " + filename);
  }

  std::uint32_t magic{};
  std::uint32_t version{};
  std::uint32_t input_planes{};
  std::uint32_t bitplanes{};
  std::uint32_t policy_size{};
  std::uint64_t example_count{};

  read_value(input, magic);
  read_value(input, version);
  read_value(input, input_planes);
  read_value(input, bitplanes);
  read_value(input, policy_size);
  read_value(input, example_count);

  if (magic != COMPACT_DATASET_MAGIC ||
      version != COMPACT_DATASET_VERSION ||
      input_planes != COMPACT_INPUT_PLANES ||
      bitplanes != COMPACT_BITPLANES || policy_size != COMPACT_POLICY_SIZE) {
    throw std::runtime_error("Unsupported compact training-data header");
  }

  if (example_count > std::numeric_limits<std::size_t>::max()) {
    throw std::runtime_error("Compact record count is too large");
  }

  std::vector<CompactTrainingExample> examples;
  examples.reserve(static_cast<std::size_t>(example_count));

  for (std::uint64_t i = 0; i < example_count; ++i) {
    CompactTrainingExample example{};
    std::uint8_t source{};
    std::uint8_t reserved{};
    std::uint16_t policy_count{};

    read_value(input, example.game_id);
    read_value(input, example.ply);
    read_value(input, source);
    read_value(input, example.castling_rights);
    read_value(input, example.side_to_move);
    read_value(input, reserved);
    read_value(input, example.rule50_count);
    read_value(input, policy_count);

    example.source = static_cast<TrainingSource>(source);

    for (std::uint64_t &plane : example.planes) {
      read_value(input, plane);
    }
    for (float &value : example.wdl) {
      read_value(input, value);
    }
    read_value(input, example.moves_left);

    example.policy.resize(policy_count);
    for (CompactPolicyEntry &entry : example.policy) {
      read_value(input, entry.index);
      read_value(input, entry.probability);
    }

    validate_example(example);
    examples.push_back(std::move(example));
  }

  return examples;
}
