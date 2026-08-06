#pragma once
#include "board.hpp"
#include <array>
#include <cstddef>
#include <sys/types.h>
#include <vector>

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
  Square from{NO_SQUARE};
  Square to{NO_SQUARE};
  MoveType type{MoveType::QUIET};
  Piece promotion_piece{NO_PIECE};

  Move() = default;

  Move(Square from_square, Square to_square)
      : from{(from_square)}, to{(to_square)} {}

  Move(Square from_square, Square to_square, MoveType type)
      : from{(from_square)}, to{(to_square)}, type{type} {}

  Move(Square from_square, Square to_square, MoveType type, Piece piece)
      : from(from_square), to{(to_square)}, type{type}, promotion_piece{piece} {
  }
  bool operator==(const Move &other) const {
    return from == other.from && to == other.to && type == other.type &&
           promotion_piece == other.promotion_piece;
  }
};

class MoveList {
private:
  static constexpr std::size_t MAX_MOVES = 256;
  std::array<Move, MAX_MOVES> moves{};
  size_t count{};

public:
  void add(Move move);
  void clear();
  std::vector<Move> get_moves() {
    return std::vector<Move>(moves.begin(), moves.begin() + count);
  }
  size_t get_count() { return count; }
  void print() const;
};
