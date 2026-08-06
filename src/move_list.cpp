#include "move_list.hpp"
#include <array>
#include <cassert>
#include <cstddef>

#include <sys/types.h>

void MoveList::add(Move move) {
  assert(count < MAX_MOVES);

  moves[count] = move;
  ++count;
}

void MoveList::clear() {
  moves.fill(Move{});
  count = 0;
}

void MoveList::print() const {
  for (std::size_t i = 0; i < count; ++i) {
    std::cout << moves[i].from << " to " << moves[i].to << '\n';
  }
}