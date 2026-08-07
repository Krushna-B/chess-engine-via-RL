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

// Forward declarations break the include cycle with move_list.hpp /
// move_generator.hpp: Position only uses Move by reference and calls
// is_square_attacked, so it needs the names, not the full definitions.
struct Move;
class Position;
bool is_square_attacked(const Position &position, Square square,
                        Side attacking_side);

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
  std::uint16_t halfmove_clock{0};
  std::uint16_t fullmove_number{1};
  std::uint8_t castling_rights{WHITE_KINGSIDE | WHITE_QUEENSIDE |
                               BLACK_KINGSIDE | BLACK_QUEENSIDE};

  void update_occupancies();
  Piece get_piece_on_square(Square square, Side side);
  void update_castling_rights_for_move(Piece moving_piece, Side moving_side,
                                       Square from, Piece captured_piece,
                                       Square captured_square);

public:
  u64 hash() const;
  Bitboard get_piece(Side side, Piece piece) const;
  Bitboard get_occupancy(Side side) const;
  Bitboard get_all_occupancy() const;
  Side get_side_to_move() const;
  Bitboard get_enimies(Side color) const;
  std::uint16_t get_halfmove_clock() const;
  std::uint16_t get_fullmove_number() const;

  // Insufficent material
  bool has_insufficent_material();
  // En Passant APIs
  Square get_en_passant_square() const;
  void set_en_passant_square(Square square);
  // Castling APIs
  bool has_castling_rights(CastlingRight right) const;
  void remove_castling_rights(CastlingRight right);
  void clear_castling_rights();
  // Starting Postion
  void set_starting_position();
  void print_position() const;
  bool is_in_check(Side side);
  // Terminal state checks (side to move has no legal moves / drawn material)
  bool is_checkmate();
  bool is_stalemate();
  bool is_draw();
  // Designing the Make move function to make moves on the board
  bool make_move(const Move &move);
  // FEN input
  bool set_from_fen(const std::string &fen);
};
