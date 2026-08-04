#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include <iostream>
#include <ostream>

// Aliaes
using Bitboard = uint64_t;
using u64 = uint64_t;
using u32 = int32_t;

// Bit Ops
#define get_bit(board, square) (board & (1ULL << square))
#define set_bit(board, square) (board |= (1ULL << square))
#define pop_bit(board, square) (board &= ~(1ULL << square))

// Pieces
enum Piece : std::uint8_t {
  PAWN,
  KNIGHT,
  BISHOP,
  ROOK,
  QUEEN,
  KING,

  NO_PIECE
};

enum Side { WHITE, BLACK };
enum Square {
  a1,
  b1,
  c1,
  d1,
  e1,
  f1,
  g1,
  h1,
  a2,
  b2,
  c2,
  d2,
  e2,
  f2,
  g2,
  h2,
  a3,
  b3,
  c3,
  d3,
  e3,
  f3,
  g3,
  h3,
  a4,
  b4,
  c4,
  d4,
  e4,
  f4,
  g4,
  h4,
  a5,
  b5,
  c5,
  d5,
  e5,
  f5,
  g5,
  h5,
  a6,
  b6,
  c6,
  d6,
  e6,
  f6,
  g6,
  h6,
  a7,
  b7,
  c7,
  d7,
  e7,
  f7,
  g7,
  h7,
  a8,
  b8,
  c8,
  d8,
  e8,
  f8,
  g8,
  h8,

  NO_SQUARE
};
// Stores whether each type of castling is possible
enum CastlingRight : std::uint8_t {
  WHITE_KINGSIDE = 1 << 0,
  WHITE_QUEENSIDE = 1 << 1,
  BLACK_KINGSIDE = 1 << 2,
  BLACK_QUEENSIDE = 1 << 3
};

// Masks

//  Ranks
constexpr Bitboard RANK_1 = 0x00000000000000FFULL;
constexpr Bitboard RANK_2 = 0x000000000000FF00ULL;
constexpr Bitboard RANK_3 = 0x0000000000FF0000ULL;
constexpr Bitboard RANK_4 = 0x00000000FF000000ULL;
constexpr Bitboard RANK_5 = 0x000000FF00000000ULL;
constexpr Bitboard RANK_6 = 0x0000FF0000000000ULL;
constexpr Bitboard RANK_7 = 0x00FF000000000000ULL;
constexpr Bitboard RANK_8 = 0xFF00000000000000ULL;

// Files
constexpr Bitboard FILE_A = 0x0101010101010101ULL;
constexpr Bitboard FILE_B = 0x0202020202020202ULL;
constexpr Bitboard FILE_C = 0x0404040404040404ULL;
constexpr Bitboard FILE_D = 0x0808080808080808ULL;
constexpr Bitboard FILE_E = 0x1010101010101010ULL;
constexpr Bitboard FILE_F = 0x2020202020202020ULL;
constexpr Bitboard FILE_G = 0x4040404040404040ULL;
constexpr Bitboard FILE_H = 0x8080808080808080ULL;

constexpr Bitboard FILE_GH = FILE_G | FILE_H;
constexpr Bitboard FILE_AB = FILE_A | FILE_B;
// Movement Ops
constexpr Bitboard west(Bitboard b) { return (b & ~FILE_A) >> 1; }
constexpr Bitboard east(Bitboard b) { return (b & ~FILE_H) << 1; }
constexpr Bitboard north(Bitboard b) { return (b & ~RANK_8) << 8; }
constexpr Bitboard south(Bitboard b) { return (b & ~RANK_1) >> 8; }
constexpr Bitboard north_west(Bitboard b) { return (b & ~FILE_A) << 7; }
constexpr Bitboard north_east(Bitboard b) { return (b & ~FILE_H) << 9; }
constexpr Bitboard south_west(Bitboard b) { return (b & ~FILE_A) >> 9; }
constexpr Bitboard south_east(Bitboard b) { return (b & ~FILE_H) >> 7; }

// I/O
inline void printBitBoard(Bitboard b) {
  std::cout << "\n";
  for (int rank{7}; rank >= 0; rank--) {
    std::cout << " " << (rank + 1) << " ";
    for (int file{}; file < 8; file++) {
      int square = rank * 8 + file;
      std::cout << " " << (get_bit(b, square) ? 1 : 0);
    }
    std::cout << "\n";
  }
  std::cout << "\n    a b c d e f g h \n\n";
}

// Overloaded operator for printing square enum types
inline std::ostream &operator<<(std::ostream &out, Square square) {
  if (square == NO_SQUARE) {
    return out << "NO_SQUARE";
  }

  const int value = static_cast<int>(square);

  if (value < 0 || value >= 64) {
    return out << "INVALID_SQUARE";
  }

  const char file = static_cast<char>('a' + value % 8);
  const char rank = static_cast<char>('1' + value / 8);

  return out << file << rank;
}

/**
Board Class
*/
class Position {
private:
  std::array<std::array<Bitboard, 7>, 2> pieces{};
  Bitboard white_occupancy{};
  Bitboard black_occupancy{};
  Bitboard all_occupancy{};
  Side side_to_move{WHITE};
  Square en_passant_square{NO_SQUARE};
  std::uint8_t castling_rights{WHITE_KINGSIDE | WHITE_QUEENSIDE |
                               BLACK_KINGSIDE | BLACK_QUEENSIDE};

public:
  Bitboard get_piece(Side side, Piece piece) const {
    return pieces[side][piece];
  }
  Bitboard get_occupancy(Side side) const {
    if (side == WHITE) {
      return white_occupancy;
    } else {
      return black_occupancy;
    }
  }

  Bitboard get_all_occupancy() const { return all_occupancy; }
  Side get_side_to_move() const { return side_to_move; }
  Bitboard get_enimies(Side color) const {
    return color == WHITE ? black_occupancy : white_occupancy;
  }
  // En Passant APIs
  Square get_en_passant_square() const { return en_passant_square; }

  void set_en_passant_square(Square square) { en_passant_square = square; }
  // Castling APIs
  bool has_castling_rights(CastlingRight right) {
    return (castling_rights & right) != 0;
  }
  void remove_castling_rights(CastlingRight right) {
    castling_rights &= ~(1 << right);
  }
  void clear_castling_rights() { castling_rights = 0; }

  //--------------
  // Starting Postion
  void set_starting_position() {
    pieces = {};
    // White Pieces

    pieces[WHITE][PAWN] = 0x000000000000FF00ULL;

    pieces[WHITE][KNIGHT] = 0x0000000000000042ULL;

    pieces[WHITE][BISHOP] = 0x0000000000000024ULL;

    pieces[WHITE][ROOK] = 0x0000000000000081ULL;

    pieces[WHITE][QUEEN] = 0x0000000000000008ULL;

    pieces[WHITE][KING] = 0x0000000000000010ULL;

    /*
     * Black pieces
     *
     * Rank 7:
     * p p p p p p p p
     *
     * Rank 8:
     * r n b q k b n r
     */
    pieces[BLACK][PAWN] = 0x00FF000000000000ULL;

    pieces[BLACK][KNIGHT] = 0x4200000000000000ULL;

    pieces[BLACK][BISHOP] = 0x2400000000000000ULL;

    pieces[BLACK][ROOK] = 0x8100000000000000ULL;

    pieces[BLACK][QUEEN] = 0x0800000000000000ULL;

    pieces[BLACK][KING] = 0x1000000000000000ULL;

    side_to_move = Side::WHITE;
    clear_castling_rights();
    update_occupancies();
    set_en_passant_square(NO_SQUARE);
  }
  void update_occupancies() {
    white_occupancy = 0;
    black_occupancy = 0;

    for (Piece piece : {PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING}) {
      white_occupancy |= pieces[WHITE][piece];
      black_occupancy |= pieces[BLACK][piece];
    }

    all_occupancy = white_occupancy | black_occupancy;
  }
  void print_position() const {
    constexpr char piece_symbols[2][6] = {{'P', 'N', 'B', 'R', 'Q', 'K'},
                                          {'p', 'n', 'b', 'r', 'q', 'k'}};

    std::cout << '\n';

    for (int rank = 7; rank >= 0; --rank) {
      std::cout << rank + 1 << "  ";

      for (int file = 0; file < 8; ++file) {
        const int square = rank * 8 + file;
        char symbol = '.';

        for (int side = WHITE; side <= BLACK; ++side) {
          for (int piece = PAWN; piece <= KING; ++piece) {
            if (get_bit(pieces[side][piece], square)) {
              symbol = piece_symbols[side][piece];
            }
          }
        }

        std::cout << symbol << ' ';
      }

      std::cout << '\n';
    }

    std::cout << "\n   a b c d e f g h\n\n";
  }
};
