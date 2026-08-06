#pragma once

#include "board.hpp"
#include "move_list.hpp"
#include <optional>
#include <random>

class RandomEngine {
private:
  std::mt19937 random_generator;

public:
  RandomEngine();

  std::optional<Move> choose_move(const Position &position);
};
