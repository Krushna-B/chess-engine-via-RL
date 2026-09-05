#include "neural_net.hpp"

/**
Turns a move and the square from [73, 64] to the linear idx in [4762]
*/
constexpr u64 policy_idx(u64 move_type, u64 original_square) {
  return move_type * BOARD_SIZE + original_square;
}
