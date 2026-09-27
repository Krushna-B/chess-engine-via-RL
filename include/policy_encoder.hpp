#pragma once

#include "move_list.hpp"

#include <cstddef>

constexpr std::size_t LC0_POLICY_SIZE = 1858;

std::size_t encode_lc0_policy_move(const Position &position,
                                   const Move &move);
Move decode_lc0_policy_move(const Position &position, std::size_t index);
