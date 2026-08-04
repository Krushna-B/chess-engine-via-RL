

// Finding the leaf nodes ( number of positions raeched during the test at some
// depth)
#include "move_list.hpp"
long nodes;

// perft driver
void perft_driver(int depth) {

  // Escape condition
  if (depth == 0) {
    // increment nodes count
    nodes++;
    return;
  }
  MoveList moves{};
  //   generate_legal_moves(moves, Position & position);
}