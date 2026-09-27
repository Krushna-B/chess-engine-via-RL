#pragma once

#include "board.hpp"
#include "position_encoder.hpp"

#include <array>
#include <sys/resource.h>

constexpr u64 BOARD_SIZE = 64;
constexpr u64 SQUARE_FEATURES = 18;
constexpr u64 WDL_SIZE = 3;

constexpr u64 POLICY_SIZE = 1858;
constexpr u64 ENCODED_STATE_SIZE = POSITION_PLANES * POSITION_PLANE_SIZE;

using PolicyArray = std::array<float, POLICY_SIZE>;
using WdlArray = std::array<float, WDL_SIZE>;

using EncodedPosition = EncodedPositionHistory;

struct NetworkOutput {
  PolicyArray policy_logits{};
  WdlArray wdl{};
  float value{};
};

u64 encode_move(const Position &position, const Move &move);
void validate_move_encoding(Position &position);
EncodedPosition encode_position(const Position &position);
