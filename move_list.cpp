#include "move_list.hpp"
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>

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
