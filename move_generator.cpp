#include "attacks.hpp"
#include "board.hpp"
#include "move_list.hpp"
#include <bit>

void generate_legal_moves(MoveList &moves, Position &position);

void generate_all_pseudo_moves(MoveList &moves, Position &position);

void generate_pawn_moves(MoveList &moves, const Position &position);

void generate_knight_moves(MoveList &moves, const Position &position);

void generate_bishop_moves(MoveList &moves, const Position &position);

void generate_rook_moves(MoveList &moves, const Position &position);

void generate_queen_moves(MoveList &moves, const Position &position);

void generate_king_moves(MoveList &moves, const Position &position);

void add_promotions(MoveList &moves, int from, int to, MoveType type);

void generate_castling_moves(MoveList &moves, const Position &position);

bool is_square_attacked(const Position &position, Square square,
                        Side attacking_side);

// Helper function to pop the position of first 1 bit (Whihc is position of
// piece)
int pop_lsb(Bitboard &bitboard) {
  const int square = static_cast<int>(std::countr_zero(bitboard));

  bitboard &= bitboard - 1;
  return square;
}
//  Helper to get opposite side
constexpr Side opposite_side(Side side) {
  return side == WHITE ? BLACK : WHITE;
}

// The top line function that generates all the legal moves in the position
void generate_legal_moves(MoveList &legal_moves, Position &position) {
  MoveList pseudo_moves{};
  generate_all_pseudo_moves(pseudo_moves,
                            position); // Genrate all of our pseudo moves

  // Empty legal move
  legal_moves.clear();

  const Side side = position.get_side_to_move();
  const auto &move_array = pseudo_moves.get_moves();

  for (std::size_t i = 0; i < pseudo_moves.get_count(); ++i) {
    const Move &move = move_array[i];

    Position test_position = position;

    if (!test_position.make_move(move)) {
      continue;
    }
    if (!test_position.is_in_check(side)) {
      legal_moves.add(move);
    }
  }
  return;
}

void generate_all_pseudo_moves(MoveList &moves, Position &position) {
  moves.clear();
  // Generate pawn moves
  generate_pawn_moves(moves, position);
  // Generate King Moves
  generate_king_moves(moves, position);
  // Generate Knight Moves
  generate_knight_moves(moves, position);
  // Generate Bishop Moves
  generate_bishop_moves(moves, position);
  // Generate Rook Moves
  generate_rook_moves(moves, position);
  // Generate Queen Moves
  generate_queen_moves(moves, position);
}

void generate_pawn_moves(MoveList &moves, const Position &position) {
  const Side side = position.get_side_to_move();
  Bitboard pawns = position.get_piece(side, PAWN);
  Bitboard enimies = position.get_enimies(side);
  Bitboard all_occupancy = position.get_all_occupancy();

  const int direction = side == WHITE ? 8 : -8;
  const int starting_rank = side == WHITE ? 1 : 6;  // This is 0 based so 0 to 7
  const int promotion_rank = side == WHITE ? 6 : 1; // This is 0 based so 0 to 7

  while (pawns != 0ULL) {
    auto from = pop_lsb(pawns);
    const int rank = from / 8;
    const int one_step = from + direction;
    const Bitboard one_step_bit = 1ULL << one_step;

    // Genreate single pawn pushes square infront must be empty
    if ((all_occupancy & one_step_bit) == 0ULL) {
      // If we are on the promotion rank then we can do a promotion type move
      // turn into knight, bishop, rook, queen
      if (rank == promotion_rank) {
        add_promotions(moves, from, one_step, MoveType::PROMOTION);
      } else {
        moves.add(Move{static_cast<Square>(from), static_cast<Square>(one_step),
                       MoveType::QUIET});
      }

      // If square infront is empty and starting then possible to do double
      //  Generate double pawn pushes
      if (rank == starting_rank) {
        const int two_steps = from + (2 * direction);
        const Bitboard two_steps_bit = 1ULL << two_steps;
        if ((all_occupancy & two_steps_bit) == 0ULL) {
          moves.add(Move{static_cast<Square>(from),
                         static_cast<Square>(two_steps),
                         MoveType::DOUBLE_PAWN_PUSH});
        }
      }
    }

    // Generate captures
    auto pawn_attacks = get_pawn_attacks(from, side) & enimies;
    while (pawn_attacks != 0ULL) {
      auto to = pop_lsb(pawn_attacks);
      // If on promotion rank we can do a promotion capture
      if (rank == promotion_rank) {
        add_promotions(moves, from, to, MoveType::PROMOTION_CAPTURE);
      } else {
        moves.add(Move{static_cast<Square>(from), static_cast<Square>(to),
                       MoveType::CAPTURE});
      }
    }

    // Generate en passant
    const int en_passant_square = position.get_en_passant_square();

    if (en_passant_square != -1) {
      const Bitboard en_passant_bit = 1ULL << en_passant_square;

      if ((get_pawn_attacks(from, side) & en_passant_bit) != 0ULL) {
        moves.add(Move{static_cast<Square>(from),
                       static_cast<Square>(en_passant_square),
                       MoveType::EN_PASSANT});
      }
    }
  }
}

void generate_knight_moves(MoveList &moves, const Position &position) {
  const Side side = position.get_side_to_move();
  Bitboard knights = position.get_piece(side, KNIGHT);
  Bitboard friendlies = position.get_occupancy(side);
  Bitboard enimies = position.get_enimies(side);
  while (knights != 0ULL) {
    auto from = pop_lsb(knights);
    auto knight_attacks = get_knight_attacks(from) & ~friendlies;

    while (knight_attacks != 0ULL) {
      auto to = pop_lsb(knight_attacks);
      MoveType type =
          get_bit(enimies, to) ? MoveType::CAPTURE : MoveType::QUIET;
      moves.add(Move{static_cast<Square>(from), static_cast<Square>(to), type});
    }
  }
}

void generate_bishop_moves(MoveList &moves, const Position &position) {
  // TODO:
  const Side side = position.get_side_to_move();
  Bitboard bishops = position.get_piece(side, BISHOP);
  Bitboard friendlies = position.get_occupancy(side);
  Bitboard all_occupancy = position.get_all_occupancy();
  Bitboard enimies = position.get_enimies(side);
  while (bishops != 0ULL) {
    auto from = pop_lsb(bishops);
    auto bishop_attacks = get_bishop_attacks(from, all_occupancy) & ~friendlies;

    while (bishop_attacks != 0ULL) {
      auto to = pop_lsb(bishop_attacks);
      MoveType type =
          get_bit(enimies, to) ? MoveType::CAPTURE : MoveType::QUIET;
      moves.add(Move{static_cast<Square>(from), static_cast<Square>(to), type});
    }
  }
}

void generate_rook_moves(MoveList &moves, const Position &position) {

  const Side side = position.get_side_to_move();
  Bitboard rooks = position.get_piece(side, ROOK);
  Bitboard friendlies = position.get_occupancy(side);
  Bitboard all_occupancy = position.get_all_occupancy();
  Bitboard enimies = position.get_enimies(side);

  while (rooks != 0ULL) {
    auto from = pop_lsb(rooks);
    auto rook_attacks = get_rook_attacks(from, all_occupancy) & ~friendlies;

    while (rook_attacks != 0ULL) {
      auto to = pop_lsb(rook_attacks);
      MoveType type =
          get_bit(enimies, to) ? MoveType::CAPTURE : MoveType::QUIET;
      moves.add(Move{static_cast<Square>(from), static_cast<Square>(to), type});
    }
  }
}

void generate_queen_moves(MoveList &moves, const Position &position) {

  const Side side = position.get_side_to_move();
  Bitboard queen = position.get_piece(side, QUEEN);
  Bitboard friendlies = position.get_occupancy(side);
  Bitboard all_occupancy = position.get_all_occupancy();
  Bitboard enimies = position.get_enimies(side);
  while (queen != 0ULL) {
    auto from = pop_lsb(queen);
    auto queen_attacks = get_queen_attacks(from, all_occupancy) & ~friendlies;

    while (queen_attacks != 0ULL) {
      auto to = pop_lsb(queen_attacks);
      MoveType type =
          get_bit(enimies, to) ? MoveType::CAPTURE : MoveType::QUIET;
      moves.add(Move{static_cast<Square>(from), static_cast<Square>(to), type});
    }
  }
}

void generate_king_moves(MoveList &moves, const Position &position) {
  const Side side = position.get_side_to_move();
  Bitboard king = position.get_piece(side, KING);
  Bitboard friendlies = position.get_occupancy(side);
  Bitboard enimies = position.get_enimies(side);
  while (king != 0ULL) {
    auto from = pop_lsb(king);
    auto king_attacks = get_king_attacks(from) & ~friendlies;

    while (king_attacks != 0ULL) {
      auto to = pop_lsb(king_attacks);
      MoveType type =
          get_bit(enimies, to) ? MoveType::CAPTURE : MoveType::QUIET;
      moves.add(Move{static_cast<Square>(from), static_cast<Square>(to), type});
    }
  }

  generate_castling_moves(moves, position);
}

void generate_castling_moves(MoveList &moves, const Position &position) {
  const Side side = position.get_side_to_move();
  const Side enemy_side = opposite_side(side);

  if (side == Side::WHITE) {
    // White king side castling #king e1 to g1, rook from h1 to f1
    if (position.has_castling_rights(CastlingRight::WHITE_KINGSIDE)) {
      bool king_pos = get_bit(position.get_piece(side, KING), Square::e1);
      bool rook_pos = get_bit(position.get_piece(side, ROOK), Square::h1);
      bool squares_empty = !get_bit(position.get_all_occupancy(), f1) &&
                           !get_bit(position.get_all_occupancy(), g1);
      bool squares_safe = !is_square_attacked(position, e1, enemy_side) &&
                          !is_square_attacked(position, f1, enemy_side) &&
                          !is_square_attacked(position, g1, enemy_side);
      if (king_pos && rook_pos && squares_empty && squares_safe) {
        moves.add(Move{e1, g1, MoveType::KING_CASTLE});
      }
    }
    // White queen side castling #king e1 to c1, rook from a1 to d1
    if (position.has_castling_rights(CastlingRight::WHITE_QUEENSIDE)) {
      bool king_pos = get_bit(position.get_piece(side, KING), Square::e1);
      bool rook_pos = get_bit(position.get_piece(side, ROOK), Square::a1);
      bool squares_empty = !get_bit(position.get_all_occupancy(), d1) &&
                           !get_bit(position.get_all_occupancy(), c1) &&
                           !get_bit(position.get_all_occupancy(), b1);
      bool squares_safe = !is_square_attacked(position, e1, enemy_side) &&
                          !is_square_attacked(position, d1, enemy_side) &&
                          !is_square_attacked(position, c1, enemy_side) &&
                          !is_square_attacked(position, b1, enemy_side);
      if (king_pos && rook_pos && squares_empty && squares_safe) {
        moves.add(Move{e1, c1, MoveType::QUEEN_CASTLE});
      }
    }
  } else {
    // Black king side castling #king e8 to g8, rook from h8 to f8
    if (position.has_castling_rights(CastlingRight::BLACK_KINGSIDE)) {
      bool king_pos = get_bit(position.get_piece(side, KING), Square::e8);
      bool rook_pos = get_bit(position.get_piece(side, ROOK), Square::h8);
      bool squares_empty = !get_bit(position.get_all_occupancy(), f8) &&
                           !get_bit(position.get_all_occupancy(), g8);
      bool squares_safe = !is_square_attacked(position, e8, enemy_side) &&
                          !is_square_attacked(position, f8, enemy_side) &&
                          !is_square_attacked(position, g8, enemy_side);
      if (king_pos && rook_pos && squares_empty && squares_safe) {
        moves.add(Move{e8, g8, MoveType::KING_CASTLE});
      }
    }
    // Black queen side castling #king e8 to c8, rook from a8 to d8
    if (position.has_castling_rights(CastlingRight::BLACK_QUEENSIDE)) {
      bool king_pos = get_bit(position.get_piece(side, KING), Square::e8);
      bool rook_pos = get_bit(position.get_piece(side, ROOK), Square::a8);
      bool squares_empty = !get_bit(position.get_all_occupancy(), d8) &&
                           !get_bit(position.get_all_occupancy(), c8) &&
                           !get_bit(position.get_all_occupancy(), b8);
      bool squares_safe = !is_square_attacked(position, e8, enemy_side) &&
                          !is_square_attacked(position, d8, enemy_side) &&
                          !is_square_attacked(position, c8, enemy_side) &&
                          !is_square_attacked(position, b8, enemy_side);
      if (king_pos && rook_pos && squares_empty && squares_safe) {
        moves.add(Move{e8, c8, MoveType::QUEEN_CASTLE});
      }
    }
  }
}

void add_promotions(MoveList &moves, int from, int to, MoveType type) {
  moves.add(
      Move{static_cast<Square>(from), static_cast<Square>(to), type, QUEEN});
  moves.add(
      Move{static_cast<Square>(from), static_cast<Square>(to), type, ROOK});
  moves.add(
      Move{static_cast<Square>(from), static_cast<Square>(to), type, BISHOP});
  moves.add(
      Move{static_cast<Square>(from), static_cast<Square>(to), type, KNIGHT});
}

// Castling Helpers

// See if a square is under attack
bool is_square_attacked(const Position &position, Square square,
                        Side attacking_side) {
  const int square_idx = static_cast<int>(square);
  const Bitboard occupancy = position.get_all_occupancy();

  if (get_pawn_attacks(square_idx, opposite_side(attacking_side)) &
      position.get_piece(attacking_side, PAWN)) {
    return true;
  }

  if (get_king_attacks(square_idx) & position.get_piece(attacking_side, KING)) {
    return true;
  }
  if (get_knight_attacks(square_idx) &
      position.get_piece(attacking_side, KNIGHT)) {
    return true;
  }
  if (get_bishop_attacks(square_idx, occupancy) &
      position.get_piece(attacking_side, BISHOP)) {
    return true;
  }
  if (get_rook_attacks(square_idx, occupancy) &
      position.get_piece(attacking_side, ROOK)) {
    return true;
  }
  if (get_queen_attacks(square_idx, occupancy) &
      position.get_piece(attacking_side, QUEEN)) {
    return true;
  }
  return false;
}
