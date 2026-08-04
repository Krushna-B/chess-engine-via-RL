#include "attacks.hpp"
#include "board.hpp"
#include "move_list.hpp"
#include <bit>

void generate_pawn_moves(MoveList &moves, const Position &position);

void generate_knight_moves(MoveList &moves, const Position &position);

void generate_bishop_moves(MoveList &moves, const Position &position);

void generate_rook_moves(MoveList &moves, const Position &position);

void generate_queen_moves(MoveList &moves, const Position &position);

void generate_king_moves(MoveList &moves, const Position &position);

void add_promotions(MoveList &moves, int from, int to, MoveType type);

void generate_castling_moves(MoveList &moves, const Position &position);

// Helper function to pop the position of first 1 bit (Whihc is position of
// piece)
int pop_lsb(Bitboard &bitboard) {
  const int square = static_cast<int>(std::countr_zero(bitboard));

  bitboard &= bitboard - 1;
  return square;
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
        moves.add(Move{from, one_step, MoveType::QUIET});
        moves.add(Move{from, one_step});
      }

      // If square infront is empty and starting then possible to do double
      //  Generate double pawn pushes
      if (rank == starting_rank) {
        const int two_steps = from + (2 * direction);
        const Bitboard two_steps_bit = 1ULL << two_steps;
        if ((all_occupancy & two_steps_bit) == 0ULL) {
          moves.add(Move{from, two_steps});
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
        moves.add(Move{from, to, MoveType::CAPTURE});
      }
    }

    // // Generate en passant
    // const int en_passant_square = position.get_en_passant_square();

    // if (en_passant_square != -1) {
    //   const Bitboard en_passant_bit = 1ULL << en_passant_square;

    //   if ((get_pawn_attacks(from, side) & en_passant_bit) != 0ULL) {
    //     moves.add(Move{from, en_passant_square, MoveType::EN_PASSANT});
    //   }
    // }
  }
}

void generate_knight_moves(MoveList &moves, const Position &position) {
  const Side side = position.get_side_to_move();
  Bitboard knights = position.get_piece(side, KNIGHT);
  Bitboard friendlies = position.get_occupancy(side);

  while (knights != 0ULL) {
    auto from = pop_lsb(knights);
    auto knight_attacks = get_knight_attakcs(from) & ~friendlies;

    while (knight_attacks != 0ULL) {
      auto to = pop_lsb(knight_attacks);
      moves.add(Move{from, to});
    }
  }
}

void generate_bishop_moves(MoveList &moves, const Position &position) {
  // TODO:
  const Side side = position.get_side_to_move();
  Bitboard bishops = position.get_piece(side, BISHOP);
  Bitboard friendlies = position.get_occupancy(side);
  Bitboard all_occupancy = position.get_all_occupancy();

  while (bishops != 0ULL) {
    auto from = pop_lsb(bishops);
    auto bishop_attacks = get_bishop_attacks(from, all_occupancy) & ~friendlies;

    while (bishop_attacks != 0ULL) {
      auto to = pop_lsb(bishop_attacks);
      moves.add(Move{from, to});
    }
  }
}

void generate_rook_moves(MoveList &moves, const Position &position) {

  const Side side = position.get_side_to_move();
  Bitboard rooks = position.get_piece(side, ROOK);
  Bitboard friendlies = position.get_occupancy(side);
  Bitboard all_occupancy = position.get_all_occupancy();

  while (rooks != 0ULL) {
    auto from = pop_lsb(rooks);
    auto rook_attacks = get_rook_attacks(from, all_occupancy) & ~friendlies;

    while (rook_attacks != 0ULL) {
      auto to = pop_lsb(rook_attacks);
      moves.add(Move{from, to});
    }
  }
}

void generate_queen_moves(MoveList &moves, const Position &position) {

  const Side side = position.get_side_to_move();
  Bitboard queen = position.get_piece(side, QUEEN);
  Bitboard friendlies = position.get_occupancy(side);
  Bitboard all_occupancy = position.get_all_occupancy();

  while (queen != 0ULL) {
    auto from = pop_lsb(queen);
    auto queen_attacks = get_queen_attacks(from, all_occupancy) & ~friendlies;

    while (queen_attacks != 0ULL) {
      auto to = pop_lsb(queen_attacks);
      moves.add(Move{from, to});
    }
  }
}

void generate_king_moves(MoveList &moves, const Position &position) {
  const Side side = position.get_side_to_move();
  Bitboard king = position.get_piece(side, KING);
  Bitboard friendlies = position.get_occupancy(side);

  while (king != 0ULL) {
    auto from = pop_lsb(king);
    auto king_attacks = get_king_attacks(from) & ~friendlies;

    while (king_attacks != 0ULL) {
      auto to = pop_lsb(king_attacks);
      moves.add(Move{from, to});
    }
  }

  //   generate_castling_moves(moves, position);
}
// generate_castling_moves(moves, position) {}

void add_promotions(MoveList &moves, int from, int to, MoveType type) {
  moves.add(Move{from, to, type, QUEEN});
  moves.add(Move{from, to, type, ROOK});
  moves.add(Move{from, to, type, BISHOP});
  moves.add(Move{from, to, type, KNIGHT});
}