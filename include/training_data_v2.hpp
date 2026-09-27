#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

constexpr std::uint32_t COMPACT_DATASET_MAGIC = 0x43485A32;
constexpr std::uint32_t COMPACT_DATASET_VERSION = 2;
constexpr std::size_t COMPACT_INPUT_PLANES = 112;
constexpr std::size_t COMPACT_BITPLANES = 104;
constexpr std::size_t COMPACT_POLICY_SIZE = 1858;

enum class TrainingSource : std::uint8_t {
  LC0 = 0,
  SELF_PLAY = 1,
};

struct CompactPolicyEntry {
  std::uint16_t index{};
  float probability{};
};

struct CompactTrainingExample {
  std::array<std::uint64_t, COMPACT_BITPLANES> planes{};
  std::uint8_t castling_rights{};
  std::uint8_t side_to_move{};
  std::uint16_t rule50_count{};
  std::vector<CompactPolicyEntry> policy{};
  std::array<float, 3> wdl{};
  float moves_left{};
  std::uint64_t game_id{};
  std::uint32_t ply{};
  TrainingSource source{TrainingSource::SELF_PLAY};
};

class CompactTrainingDataWriter {
public:
  explicit CompactTrainingDataWriter(const std::string &filename);
  ~CompactTrainingDataWriter();
  CompactTrainingDataWriter(const CompactTrainingDataWriter &) = delete;
  CompactTrainingDataWriter &operator=(const CompactTrainingDataWriter &) = delete;

  void append(const std::vector<CompactTrainingExample> &examples);
  void close();
  std::uint64_t example_count() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

void save_compact_training_examples(
    const std::string &filename,
    const std::vector<CompactTrainingExample> &examples);

std::vector<CompactTrainingExample>
load_compact_training_examples(const std::string &filename);
