#pragma once

#include "board.hpp"

#include <vector>

class PositionHistory {
private:
  std::vector<Position> positions{};
  std::vector<u64> position_hashes{};

public:
  void add_position(const Position &position) {
    positions.push_back(position);
    position_hashes.push_back(position.hash());
  }

  void clear() {
    positions.clear();
    position_hashes.clear();
  }

  std::size_t size() const { return positions.size(); }

  const Position &get_position(std::size_t index) const {
    return positions.at(index);
  }

  int count_position(u64 position_hash) const {
    int repetitions = 0;

    for (u64 hash : position_hashes) {
      if (hash == position_hash) {
        ++repetitions;
      }
    }

    return repetitions;
  }

  bool is_threefold_repetition() const {
    if (position_hashes.empty()) {
      return false;
    }

    return count_position(position_hashes.back()) >= 3;
  }
};
