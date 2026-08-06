#include "random_engine.hpp"
#include "board.hpp"
#include "move_generator.hpp"
#include "move_list.hpp"
#include <optional>
#include <random>

RandomEngine::RandomEngine() : random_generator(std::random_device{}()) {}

std::optional<Move> RandomEngine::choose_move(const Position &position) {
  MoveList move_list{};

  Position position_copy = position;
  generate_legal_moves(move_list, position_copy);
  auto moves = move_list.get_moves();

  if (moves.size() == 0) {
    return std::nullopt;
  }

  // Select a random possible move
  std::uniform_int_distribution<int> distribution(0, moves.size() - 1);

  int random_index = distribution(random_generator);

  return moves[random_index];
}
