#pragma once

#include "board.hpp"

#include <vector>

class PositionHistory {
private:
  std::vector<u64> positions{};

public:
  void add_position(u64 position_hash) { positions.push_back(position_hash); }

  void clear() { positions.clear(); }

  int count_position(u64 position_hash) const {
    int repetitions = 0;

    for (u64 position : positions) {
      if (position == position_hash) {
        ++repetitions;
      }
    }

    return repetitions;
  }

  bool is_threefold_repetition() const {
    if (positions.empty()) {
      return false;
    }

    return count_position(positions.back()) >= 3;
  }
};
