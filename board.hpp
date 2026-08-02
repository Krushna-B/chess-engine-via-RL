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
enum Piece {
  WHITE_PAWN,
  WHITE_KNIGHT,
  WHITE_BISHOP,
  WHITE_ROOK,
  WHITE_QUEEN,
  WHITE_KING,

  BLACK_PAWN,
  BLACK_KNIGHT,
  BLACK_BISHOP,
  BLACK_ROOK,
  BLACK_QUEEN,
  BLACK_KING
};

enum Color { WHITE, BLACK };
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

/**
Board Class
*/
class Position {
private:
  std::array<std::array<Bitboard, 6>, 2> pieces{};
  Bitboard white_occupancy{};
  Bitboard black_occupancy{};
  Bitboard all_occupancy{};

public:
  Bitboard get_pieces(Color color, Piece piece) { return pieces[color][piece]; }
};
