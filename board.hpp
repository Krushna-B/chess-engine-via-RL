#include <cstdint>
#include <cstdio>
#include <iostream>
#include <ostream>

// Aliaes
using Bitboard = uint64_t;
using u64 = uint64_t;
using u32 = int32_t;

// Bit Ops
#define get_bit(board, square) (board & (1ULL << square))
#define set_bit(board, square) (board |= (1ULL << square))
#define clear_bit(board, square) (board &= (0ULL << square))

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

// Movement Ops
constexpr Bitboard west(Bitboard b) { return (b & ~FILE_A) >> 1; }
constexpr Bitboard east(Bitboard b) { return (b & ~FILE_H) << 1; }
constexpr Bitboard north(Bitboard b) { return (b & ~RANK_8) << 8; }
constexpr Bitboard south(Bitboard b) { return (b & ~RANK_1) >> 8; }
constexpr Bitboard north_west(Bitboard b) { return (b & ~FILE_A) << 7; }
constexpr Bitboard north_east(Bitboard b) { return (b & ~FILE_H) << 9; }
constexpr Bitboard south_west(Bitboard b) { return (b & ~FILE_A) << 9; }
constexpr Bitboard south_east(Bitboard b) { return (b & ~FILE_H) >> 7; }

class Board {
private:
  Bitboard pieces[12];
  bool turn;
  u32 moveCounter;

public:
  Board() { pieces[0] = 0x000000000000FF00ULL; }

  void printBitBoard(Bitboard &b) {
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

  void printBoard() {}

  Bitboard getBitboard(int p) { return pieces[p]; }
};
