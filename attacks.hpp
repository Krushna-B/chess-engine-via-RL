#pragma once
#include "board.hpp"
#include "magic_nums.hpp"
#include <random>

constexpr Bitboard mask_bishop_attcks(int square);

constexpr Bitboard bishop_attacks_on_the_fly(int square, Bitboard blockers);

constexpr Bitboard mask_rook_attacks(int square);

constexpr Bitboard rook_attacks_on_the_fly(int square, Bitboard blockers);

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
inline constexpr auto PAWN_LUT = generatePawnLUT(); // [Color][Square]

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

/**
Magic Number genreation template
*/
template <std::size_t MAX_OCCUPANCIES, typename MaskFunction,
          typename AttackFunction>
inline Bitboard find_magic(int square, MaskFunction mask_function,
                           AttackFunction attack_function) {
  const Bitboard attack_mask = mask_function(square);
  const int relevant_bits = std::popcount(attack_mask);
  const int occupancy_count = 1 << relevant_bits;

  // Store all occupancies and all attacks
  std::array<Bitboard, MAX_OCCUPANCIES> occupancies{};
  std::array<Bitboard, MAX_OCCUPANCIES> attacks{};

  // For each occupancy generate that occupany and it's attack
  for (int idx{}; idx < occupancy_count; idx++) {
    occupancies[idx] = set_occupancy(idx, relevant_bits, attack_mask);
    attacks[idx] = attack_function(square, occupancies[idx]);
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
    std::array<Bitboard, MAX_OCCUPANCIES> used_attacks{};
    std::array<bool, MAX_OCCUPANCIES> filled_occupanies{};
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
// For both bishop and rook
inline std::pair<std::array<Bitboard, 64>, std::array<Bitboard, 64>>
generate_all_magics() {
  std::array<Bitboard, 64> bishop_magics{};
  std::array<Bitboard, 64> rook_magics{};
  for (int square{}; square < 64; square++) {
    bishop_magics[square] = find_magic<MAX_BISHOP_OCCUPANCIES>(
        square, mask_bishop_attcks, bishop_attacks_on_the_fly);
    rook_magics[square] = find_magic<MAX_ROOK_OCCUPANCIES>(
        square, mask_rook_attacks, rook_attacks_on_the_fly);
  }
  return {bishop_magics, rook_magics};
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
------------------
Generate Rook LUT
-----------------
Helper mask for rook at each position
*/
constexpr Bitboard mask_rook_attacks(int square) {
  Bitboard mask = 0ULL;

  const int tr = square / 8;
  const int tf = square % 8;

  // North
  for (int r = tr + 1; r <= 6; ++r) {
    mask |= 1ULL << (r * 8 + tf);
  }

  // South
  for (int r = tr - 1; r >= 1; --r) {
    mask |= 1ULL << (r * 8 + tf);
  }

  // East
  for (int f = tf + 1; f <= 6; ++f) {
    mask |= 1ULL << (tr * 8 + f);
  }

  // West
  for (int f = tf - 1; f >= 1; --f) {
    mask |= 1ULL << (tr * 8 + f);
  }

  return mask;
}

// On the fly gerneation for rook attacks
constexpr Bitboard rook_attacks_on_the_fly(int square, Bitboard blockers) {
  Bitboard attacks = 0ULL;

  const int tr = square / 8;
  const int tf = square % 8;

  // North
  for (int r = tr + 1; r <= 7; ++r) {
    const Bitboard target = 1ULL << (r * 8 + tf);

    attacks |= target;

    if (blockers & target) {
      break;
    }
  }

  // South
  for (int r = tr - 1; r >= 0; --r) {
    const Bitboard target = 1ULL << (r * 8 + tf);

    attacks |= target;

    if (blockers & target) {
      break;
    }
  }

  // East
  for (int f = tf + 1; f <= 7; ++f) {
    const Bitboard target = 1ULL << (tr * 8 + f);

    attacks |= target;

    if (blockers & target) {
      break;
    }
  }

  // West
  for (int f = tf - 1; f >= 0; --f) {
    const Bitboard target = 1ULL << (tr * 8 + f);

    attacks |= target;

    if (blockers & target) {
      break;
    }
  }

  return attacks;
}

struct ROOK_TABLE {
  std::array<Bitboard, 64> ROOK_MASKS{};
  std::array<int, 64> ROOK_RELEVANT_BITS{};
  std::array<std::array<Bitboard, MAX_ROOK_OCCUPANCIES>, 64>
      ROOK_LUT{}; // LUT[square][hash_idx]
};

consteval ROOK_TABLE generateRookLUT() {
  ROOK_TABLE rook_table{};

  for (int square{}; square < 64; square++) {
    const Bitboard mask = mask_rook_attacks(square);
    const int relevant_bits = std::popcount(mask);
    const int occupancy_count = 1 << relevant_bits;

    rook_table.ROOK_MASKS[square] = mask;
    rook_table.ROOK_RELEVANT_BITS[square] = relevant_bits;

    for (int occupancy_index = 0; occupancy_index < occupancy_count;
         ++occupancy_index) {
      const Bitboard occupancy =
          set_occupancy(occupancy_index, relevant_bits, mask);

      const std::size_t hashed_index =
          magic_index(occupancy, ROOK_MAGICS[square], relevant_bits);

      rook_table.ROOK_LUT[square][hashed_index] =
          rook_attacks_on_the_fly(square, occupancy);
    }
  }
  return rook_table;
}
inline constexpr ROOK_TABLE ROOK_TABLES = generateRookLUT();

/***
---------------------------
API's to get attacks from look up tables
*/
constexpr Bitboard get_bishop_attacks(int square, Bitboard occupancy) {
  const Bitboard mask = BISHOP_TABLES.BISHOP_MASKS[square];
  const int relevant_bits = BISHOP_TABLES.BISHOP_RELEVANT_BITS[square];
  // Remove squares that are not in the bishop's attack mask
  occupancy &= mask;

  const std::size_t hash_idx =
      magic_index(occupancy, BISHOP_MAGICS[square], relevant_bits);

  return BISHOP_TABLES.BISHOP_LUT[square][hash_idx];
}

constexpr Bitboard get_rook_attacks(int square, Bitboard occupancy) {
  const Bitboard mask = ROOK_TABLES.ROOK_MASKS[square];

  const int relevant_bits = ROOK_TABLES.ROOK_RELEVANT_BITS[square];

  occupancy &= mask;

  const std::size_t hash_idx =
      magic_index(occupancy, ROOK_MAGICS[square], relevant_bits);

  return ROOK_TABLES.ROOK_LUT[square][hash_idx];
}

constexpr Bitboard get_queen_attacks(int square, Bitboard occupancy) {
  return (get_bishop_attacks(square, occupancy) |
          get_rook_attacks(square, occupancy));
}

constexpr Bitboard get_knight_attakcs(int square) { return KNIGHT_LUT[square]; }

constexpr Bitboard get_king_attacks(int square) { return KING_LUT[square]; }

constexpr Bitboard get_pawn_attacks(int square, Color color) {
  return PAWN_LUT[color][square];
}

/***
Checking if a square is attacked by the current given side
*/
inline bool is_square_attacked(int square, Color color) {
  // White Pawns
  if (color == Color::WHITE && get_pawn_attacks[Color::BLACK][square])

    return false;
}
