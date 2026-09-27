#include "policy_encoder.hpp"

#include <array>
#include <cmath>
#include <stdexcept>

namespace {

struct PolicyMove {
  Square from;
  Square to;
  Piece promotion_piece;
};

constexpr int absolute(int value) { return value < 0 ? -value : value; }

constexpr bool is_regular_policy_move(int from, int to) {
  if (from == to) {
    return false;
  }

  int rank_change = absolute(to / 8 - from / 8);
  int file_change = absolute(to % 8 - from % 8);
  bool queen_move = rank_change == 0 || file_change == 0 ||
                    rank_change == file_change;
  bool knight_move =
      (rank_change == 2 && file_change == 1) ||
      (rank_change == 1 && file_change == 2);
  return queen_move || knight_move;
}

consteval std::array<PolicyMove, LC0_POLICY_SIZE> make_policy_moves() {
  std::array<PolicyMove, LC0_POLICY_SIZE> moves{};
  std::size_t index = 0;

  for (int from = 0; from < 64; ++from) {
    for (int to = 0; to < 64; ++to) {
      if (is_regular_policy_move(from, to)) {
        moves[index++] = PolicyMove{static_cast<Square>(from),
                                    static_cast<Square>(to), NO_PIECE};
      }
    }
  }

  constexpr std::array<Piece, 3> PROMOTION_PIECES{QUEEN, ROOK, BISHOP};

  for (int from_file = 0; from_file < 8; ++from_file) {
    int first_to_file = from_file == 0 ? 0 : from_file - 1;
    int last_to_file = from_file == 7 ? 7 : from_file + 1;

    for (int to_file = first_to_file; to_file <= last_to_file; ++to_file) {
      for (Piece promotion_piece : PROMOTION_PIECES) {
        moves[index++] =
            PolicyMove{static_cast<Square>(a7 + from_file),
                       static_cast<Square>(a8 + to_file), promotion_piece};
      }
    }
  }

  return moves;
}

constexpr std::array<PolicyMove, LC0_POLICY_SIZE> POLICY_MOVES =
    make_policy_moves();

Square flip_rank(Square square) {
  return static_cast<Square>(static_cast<int>(square) ^ 56);
}

Square canonical_square(Square square, Side side_to_move) {
  return side_to_move == WHITE ? square : flip_rank(square);
}

bool is_promotion(const Move &move) {
  return move.type == MoveType::PROMOTION ||
         move.type == MoveType::PROMOTION_CAPTURE;
}

MoveType move_type(const Position &position, Square from, Square to,
                   Piece moving_piece, bool promotion) {
  Side side_to_move = position.get_side_to_move();
  Side opponent_side = side_to_move == WHITE ? BLACK : WHITE;
  Bitboard target = Bitboard{1} << static_cast<int>(to);
  bool capture = (position.get_occupancy(opponent_side) & target) != 0;

  if (promotion) {
    return capture ? MoveType::PROMOTION_CAPTURE : MoveType::PROMOTION;
  }

  if (moving_piece == KING) {
    Square canonical_from = canonical_square(from, side_to_move);
    Square canonical_to = canonical_square(to, side_to_move);

    if (canonical_from == e1 && canonical_to == g1) {
      return MoveType::KING_CASTLE;
    }
    if (canonical_from == e1 && canonical_to == c1) {
      return MoveType::QUEEN_CASTLE;
    }
  }

  if (moving_piece == PAWN) {
    if (to == position.get_en_passant_square() && !capture) {
      return MoveType::EN_PASSANT;
    }
    if (std::abs(static_cast<int>(to) - static_cast<int>(from)) == 16) {
      return MoveType::DOUBLE_PAWN_PUSH;
    }
  }

  return capture ? MoveType::CAPTURE : MoveType::QUIET;
}

Piece piece_on_square(const Position &position, Square square, Side side) {
  Bitboard target = Bitboard{1} << static_cast<int>(square);

  for (Piece piece : {PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING}) {
    if ((position.get_piece(side, piece) & target) != 0) {
      return piece;
    }
  }

  return NO_PIECE;
}

} // namespace

std::size_t encode_lc0_policy_move(const Position &position,
                                   const Move &move) {
  Side side_to_move = position.get_side_to_move();
  Square from = canonical_square(move.from, side_to_move);
  Square to = canonical_square(move.to, side_to_move);
  Piece promotion_piece = is_promotion(move) ? move.promotion_piece : NO_PIECE;

  if (promotion_piece == KNIGHT) {
    promotion_piece = NO_PIECE;
  }

  for (std::size_t index = 0; index < POLICY_MOVES.size(); ++index) {
    const PolicyMove &policy_move = POLICY_MOVES[index];

    if (policy_move.from == from && policy_move.to == to &&
        policy_move.promotion_piece == promotion_piece) {
      return index;
    }
  }

  throw std::invalid_argument("Move is not in the Lc0 policy map");
}

Move decode_lc0_policy_move(const Position &position, std::size_t index) {
  if (index >= POLICY_MOVES.size()) {
    throw std::out_of_range("Lc0 policy index is out of range");
  }

  Side side_to_move = position.get_side_to_move();
  PolicyMove policy_move = POLICY_MOVES[index];
  Square from = canonical_square(policy_move.from, side_to_move);
  Square to = canonical_square(policy_move.to, side_to_move);
  Piece moving_piece = piece_on_square(position, from, side_to_move);
  Piece promotion_piece = policy_move.promotion_piece;
  bool promotion = promotion_piece != NO_PIECE;

  if (!promotion && moving_piece == PAWN &&
      static_cast<int>(canonical_square(from, side_to_move)) / 8 == 6 &&
      static_cast<int>(canonical_square(to, side_to_move)) / 8 == 7) {
    promotion = true;
    promotion_piece = KNIGHT;
  }

  MoveType type = move_type(position, from, to, moving_piece, promotion);
  return Move{from, to, type, promotion_piece};
}
