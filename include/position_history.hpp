#pragma once

#include "board.hpp"

#include <vector>

class PositionHistory {
private:
  std::vector<u64> positions{};

public:
  void add_position(u64 position_hash) { positions.push_back(position_hash); }

  void clear() { positions.clear(); }

  bool is_threefold_repetition() const {
    if (positions.empty()) {
      return false;
    }

    u64 current_position = positions.back();
    int repetitions = 0;

    for (u64 position : positions) {
      if (position == current_position) {
        ++repetitions;
      }
    }

    return repetitions >= 3;
  }
};
