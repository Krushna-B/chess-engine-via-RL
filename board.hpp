#pragma once
#include <array>
#include <bit>
#include <cstdint>

#include "magic_nums.hpp"
#include <iostream>
#include <ostream>
#include <random>

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
-----------------------
Attacks
------------------------
 */
// Knight LUT
consteval std::array<Bitboard, 64> generateKnightLUT() {
  std::array<Bitboard, 64> lut{};
  for (auto i{0ULL}; i < 64; i++) {
    const Bitboard knight = 1ULL << i;
    lut[i] = ((knight & ~FILE_GH) << 10) | ((knight & ~FILE_AB) << 6) |
             ((knight & ~FILE_H) << 17) | ((knight & ~FILE_A) << 15) |
             ((knight & ~FILE_GH) >> 6) | ((knight & ~FILE_AB) >> 10) |
             ((knight & ~FILE_H) >> 15) | ((knight & ~FILE_A) >> 17);
  }
  return lut;
}
inline constexpr auto KNIGHT_LUT = generateKnightLUT(); // KNIGHT LUT

// King LUT
consteval std::array<Bitboard, 64> generateKingLUT() {
  std::array<Bitboard, 64> lut{};
  for (auto i{0ULL}; i < 64; i++) {
    const Bitboard king = 1ULL << i;
    lut[i] = north(king) | south(king) | east(king) | west(king) |
             north_east(king) | north_west(king) | south_east(king) |
             south_west(king);
  }
  return lut;
}
inline constexpr auto KING_LUT = generateKingLUT();

// Pawn LUT
consteval std::array<std::array<Bitboard, 64>, 2> generatePawnLUT() {
  std::array<std::array<Bitboard, 64>, 2> lut{};
  for (auto square{0ULL}; square < 64; square++) {
    const Bitboard pawn = 1ULL << square;
    // White Pieces
    lut[WHITE][square] = north_west(pawn) | north_east(pawn);

    // Black Pieces
    lut[BLACK][square] = south_west(pawn) | south_east(pawn);
  }
  return lut;
}
inline constexpr auto PAWN_ATTACKS = generatePawnLUT();

/**
---------------------------
Magic Bitboards
---------------------------
 */

constexpr int MAX_BISHOP_OCCUPANCIES = 1 << 9;
constexpr int MAX_ROOK_OCCUPANCIES = 1 << 12;

// Set occupany builder
constexpr Bitboard set_occupancy(int idx, int bits_in_mask,
                                 Bitboard attack_mask) {
  Bitboard occupancy{0ULL};

  // loop over range of bits within the attack mask
  for (int i{}; i < bits_in_mask; i++) {
    const int square = std::countr_zero(attack_mask);
    pop_bit(attack_mask, square);

    // make sure on the baord
    if (idx & (1 << i)) {
      occupancy |= 1ULL << square;
    }
  }
  return occupancy;
}

/**
Magic index generation
 */
constexpr std::size_t magic_index(Bitboard occupancy, Bitboard magic,
                                  int relevant_bits) {
  return static_cast<std::size_t>((occupancy * magic) >> (64 - relevant_bits));
}
// Not crytographically secure, psuedo RNG
inline Bitboard random_u64() {
  // Seeded via hardware entropy
  std::mt19937_64 engine{std::random_device{}()};
  return engine();
}

/***
Bishop LUT
*/
/**
Helper mask for bishop at each position
 */
constexpr Bitboard mask_bishop_attcks(int square) {
  Bitboard attacks{};
  // Init rows and targets
  int r, f;
  const int tr = square / 8;
  const int tf = square % 8;

  // North-east
  for (r = tr + 1, f = tf + 1; r <= 6 && f <= 6; r++, f++) {
    attacks |= (1ULL << (r * 8 + f));
  }

  // South-east
  for (r = tr - 1, f = tf + 1; r >= 1 && f <= 6; r--, f++) {
    attacks |= (1ULL << (r * 8 + f));
  }

  // North-west
  for (r = tr + 1, f = tf - 1; r <= 6 && f >= 1; r++, f--) {
    attacks |= (1ULL << (r * 8 + f));
  }

  // South-west
  for (r = tr - 1, f = tf - 1; r >= 1 && f >= 1; r--, f--) {
    attacks |= (1ULL << (r * 8 + f));
  }

  return attacks;
}

// On the fly genration
constexpr Bitboard bishop_attacks_on_the_fly(int square, Bitboard blockers) {
  Bitboard attacks = 0ULL;
  // Init rows and targets
  int r, f;
  const int tr = square / 8;
  const int tf = square % 8;

  // North-east
  for (r = tr + 1, f = tf + 1; r <= 7 && f <= 7; r++, f++) {
    const Bitboard target = 1ULL << ((r * 8 + f));
    attacks |= target;
    if (blockers & target) {
      break;
    }
  }

  // South-east
  for (r = tr - 1, f = tf + 1; r >= 0 && f <= 7; r--, f++) {
    const Bitboard target = 1ULL << ((r * 8 + f));
    attacks |= target;
    if (blockers & target) {
      break;
    }
  }

  // North-west
  for (r = tr + 1, f = tf - 1; r <= 7 && f >= 0; r++, f--) {
    const Bitboard target = 1ULL << ((r * 8 + f));
    attacks |= target;
    if (blockers & target) {
      break;
    }
  }

  // South-west
  for (r = tr - 1, f = tf - 1; r >= 0 && f >= 0; r--, f--) {
    const Bitboard target = 1ULL << ((r * 8 + f));
    attacks |= target;
    if (blockers & target) {
      break;
    }
  }
  return attacks;
}
inline Bitboard find_bishop_magic(int square) {
  const Bitboard attack_mask = mask_bishop_attcks(square);
  const int relevant_bits = std::popcount(attack_mask);
  const int occupancy_count = 1 << relevant_bits;

  // Store all occupancies and all attacks
  std::array<Bitboard, MAX_BISHOP_OCCUPANCIES> occupancies{};
  std::array<Bitboard, MAX_BISHOP_OCCUPANCIES> attacks{};

  // For each occupancy generate that occupany and it's attack
  for (int idx{}; idx < occupancy_count; idx++) {
    occupancies[idx] = set_occupancy(idx, relevant_bits, attack_mask);
    attacks[idx] = bishop_attacks_on_the_fly(square, occupancies[idx]);
  }

  // Find magic number candidate via trial and error
  while (true) {
    // Create a spare magic_num
    const Bitboard magic_num = random_u64() & random_u64() & random_u64();

    // Early-rejection heuristic
    if (std::popcount((attack_mask * magic_num) & 0xFF00000000000000ULL) < 6) {
      continue;
    }

    std::cout << "Trying magic num: " << magic_num << " for square " << square
              << std::endl;
    std::array<Bitboard, MAX_BISHOP_OCCUPANCIES> used_attacks{};
    std::array<bool, MAX_BISHOP_OCCUPANCIES> filled_occupanies{};
    bool failed = false;

    for (int idx{}; idx < occupancy_count; idx++) {
      auto hash_index = magic_index(occupancies[idx], magic_num, relevant_bits);
      if (!filled_occupanies[hash_index]) {
        filled_occupanies[hash_index] = true;
        used_attacks[hash_index] = attacks[idx];
      } else if (used_attacks[hash_index] != attacks[idx]) {
        failed = true;
        break;
      }
    }
    if (!failed) {
      return magic_num;
    }
  }
}
// Magic number's for every single square a1,......h8
inline std::array<Bitboard, 64> generate_all_bishop_magics() {
  std::array<Bitboard, 64> magics{};
  for (int square{}; square < 64; square++) {
    magics[square] = find_bishop_magic(square);
  }
  return magics;
}

/***
Build the Bishop LUT
 */
struct BISHOP_TABLE {
  std::array<Bitboard, 64> BISHOP_MASKS{};
  std::array<int, 64> BISHOP_RELEVANT_BITS{};
  std::array<std::array<Bitboard, MAX_BISHOP_OCCUPANCIES>, 64>
      BISHOP_LUT{}; // LUT[square][hash_idx]
};

consteval BISHOP_TABLE generateBishopLUT() {
  BISHOP_TABLE bishop_table{};

  for (int square{}; square < 64; square++) {
    const Bitboard mask = mask_bishop_attcks(square);
    const int relevant_bits = std::popcount(mask);
    const int occupancy_count = 1 << relevant_bits;

    bishop_table.BISHOP_MASKS[square] = mask;
    bishop_table.BISHOP_RELEVANT_BITS[square] = relevant_bits;

    for (int occupancy_index = 0; occupancy_index < occupancy_count;
         ++occupancy_index) {
      const Bitboard occupancy =
          set_occupancy(occupancy_index, relevant_bits, mask);

      const std::size_t hashed_index =
          magic_index(occupancy, BISHOP_MAGICS[square], relevant_bits);

      bishop_table.BISHOP_LUT[square][hashed_index] =
          bishop_attacks_on_the_fly(square, occupancy);
    }
  }
  return bishop_table;
}
inline constexpr BISHOP_TABLE BISHOP_TABLES = generateBishopLUT();

/**
Generate Rook LUT
Helper mask for rook at each position
*/
