#pragma once
#include "board.hpp"
#include <array>
#include <cstddef>
#include <sys/types.h>

enum class MoveType : std::uint8_t {
  QUIET,
  CAPTURE,
  DOUBLE_PAWN_PUSH,
  EN_PASSANT,
  PROMOTION,
  PROMOTION_CAPTURE,
  KING_CASTLE,
  QUEEN_CASTLE
};

struct Move {
  uint8_t from{};
  uint8_t to{};
  MoveType type{MoveType::QUIET};
  Piece promotion_piece{NO_PIECE};

  Move() = default;

  Move(int from_square, int to_square)
      : from{static_cast<uint8_t>(from_square)},
        to{static_cast<uint8_t>(to_square)} {}

  Move(int from_square, int to_square, MoveType type)
      : from{static_cast<uint8_t>(from_square)},
        to{static_cast<uint8_t>(to_square)}, type{type} {}

  Move(int from_square, int to_square, MoveType type, Piece piece)
      : from{static_cast<uint8_t>(from_square)},
        to{static_cast<uint8_t>(to_square)}, type{type},
        promotion_piece{piece} {}
};

class MoveList {
private:
  static constexpr std::size_t MAX_MOVES = 256;
  std::array<Move, MAX_MOVES> moves{};
  size_t count{};

public:
  void add(Move move);
  void clear();
};
