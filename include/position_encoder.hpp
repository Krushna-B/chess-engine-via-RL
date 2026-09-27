#pragma once

#include "position_history.hpp"

#include <array>
#include <cstddef>

constexpr std::size_t POSITION_PLANES = 112;
constexpr std::size_t POSITION_PLANE_SIZE = 64;
constexpr std::size_t ENCODED_POSITION_HISTORY_SIZE =
    POSITION_PLANES * POSITION_PLANE_SIZE;

using EncodedPositionHistory =
    std::array<float, ENCODED_POSITION_HISTORY_SIZE>;

EncodedPositionHistory encode_position_history(const PositionHistory &history);
