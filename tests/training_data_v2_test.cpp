#include "training_data_v2.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace {

bool check(bool condition, const char *message) {
  if (!condition) {
    std::cerr << message << '\n';
    return false;
  }

  return true;
}

bool equal(const CompactTrainingExample &left,
           const CompactTrainingExample &right) {
  return left.planes == right.planes &&
         left.castling_rights == right.castling_rights &&
         left.side_to_move == right.side_to_move &&
         left.rule50_count == right.rule50_count &&
         left.policy.size() == right.policy.size() &&
         left.wdl == right.wdl && left.moves_left == right.moves_left &&
         left.game_id == right.game_id && left.ply == right.ply &&
         left.source == right.source &&
         [&left, &right] {
           for (std::size_t i = 0; i < left.policy.size(); ++i) {
             if (left.policy[i].index != right.policy[i].index ||
                 left.policy[i].probability != right.policy[i].probability) {
               return false;
             }
           }
           return true;
         }();
}

} // namespace

int main() {
  std::filesystem::path filename =
      std::filesystem::temp_directory_path() /
      "chess_engine_training_data_v2_test.bin";

  CompactTrainingExample first{};
  first.planes[0] = 1ULL << 8;
  first.planes[103] = 1ULL << 63;
  first.castling_rights = 0x0F;
  first.side_to_move = 0;
  first.rule50_count = 12;
  first.policy = {{0, 0.25f}, {1792, 0.75f}};
  first.wdl = {0.7f, 0.2f, 0.1f};
  first.moves_left = 42.0f;
  first.game_id = 17;
  first.ply = 23;
  first.source = TrainingSource::LC0;

  CompactTrainingExample second{};
  second.planes[12] = 1ULL << 4;
  second.castling_rights = 0x05;
  second.side_to_move = 1;
  second.rule50_count = 87;
  second.policy = {{301, 1.0f}};
  second.wdl = {0.1f, 0.3f, 0.6f};
  second.moves_left = 8.0f;
  second.game_id = 18;
  second.ply = 91;
  second.source = TrainingSource::SELF_PLAY;

  std::vector<CompactTrainingExample> examples{first, second};
  save_compact_training_examples(filename.string(), examples);
  std::vector<CompactTrainingExample> loaded =
      load_compact_training_examples(filename.string());

  if (!check(loaded.size() == examples.size(),
             "Compact record count did not round trip") ||
      !check(equal(loaded[0], first),
             "First compact record did not round trip") ||
      !check(equal(loaded[1], second),
             "Second compact record did not round trip")) {
    std::filesystem::remove(filename);
    return 1;
  }

  CompactTrainingExample invalid = first;
  invalid.policy.push_back({0, 0.1f});

  try {
    save_compact_training_examples(filename.string(), {invalid});
    std::cerr << "Duplicate compact policy index was accepted\n";
    std::filesystem::remove(filename);
    return 1;
  } catch (const std::runtime_error &) {
  }

  std::filesystem::remove(filename);
  return 0;
}
