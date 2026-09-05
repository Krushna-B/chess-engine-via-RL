#pragma once

#include "board.hpp"

#include <array>
#include <sys/resource.h>

constexpr u64 BOARD_SIZE = 64;
constexpr u64 MOVE_TYPES = 73;
constexpr u64 SQUARE_FEATURES = 18;

constexpr u64 POLICY_SIZE = BOARD_SIZE * MOVE_TYPES;
constexpr u64 ENCODED_STATE_SIZE = BOARD_SIZE * SQUARE_FEATURES;

using PolicyArray = std::array<float, POLICY_SIZE>; //[64 18]

using EncodedPosition =
    std::array<float, ENCODED_STATE_SIZE>; //[73, 64] flatten to [4672]

struct NetworkOutput {
  PolicyArray policy_logits{};
  float value{};
};

u64 encode_move(const Position &position, const Move &move);
void validate_move_encoding(Position &position);
EncodedPosition encode_position(const Position &position);