#include "board.hpp"
#include "move_generator.hpp"
#include "move_list.hpp"
#include <bit>
#include <cmath>
#include <random>
#include <sstream>
#include <string>

const u64 ZOBRIST_HASH_SEED = 0x1234ABCDULL;

// Zobrist struct
struct Zobrist {
  u64 pieces[2][6][64];
  u64 castling[16];
  u64 en_passant_file[8];
  u64 side;

  Zobrist() {
    std::mt19937_64 rng(ZOBRIST_HASH_SEED);
    // Random init
    for (auto &s : pieces)
      for (auto &p : s)
        for (auto &k : p)
          k = rng();
    for (auto &k : castling)
      k = rng();
    for (auto &k : en_passant_file)
      k = rng();
    side = rng();
  }
};
const Zobrist ZOBRIST;

/**
Zobrist Hashing
*/
u64 Position::hash() const {
  u64 hash = 0;

  for (int s{}; s < 2; s++) {
    for (int p{}; p < 6; p++) {
      Bitboard b = pieces[s][p];
      while (b) {
        // For every piece
        int sq = std::countr_zero(b);
        b &= b - 1;
        hash ^= ZOBRIST.pieces[s][p][sq];
      }
    }
  }
  hash ^= ZOBRIST.castling[castling_rights & 0xF];
  if (en_passant_square != NO_SQUARE)
    hash ^= ZOBRIST.en_passant_file[en_passant_square % 8];
  if (side_to_move == BLACK)
    hash ^= ZOBRIST.side;

  return hash;
}

void Position::update_occupancies() {
  white_occupancy = 0;
  black_occupancy = 0;

  for (Piece piece : {PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING}) {
    white_occupancy |= pieces[WHITE][piece];
    black_occupancy |= pieces[BLACK][piece];
  }

  all_occupancy = white_occupancy | black_occupancy;
}

Piece Position::get_piece_on_square(Square square, Side side) {
  for (Piece piece : {PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING}) {
    if (get_bit(pieces[side][piece], square)) {
      return piece;
    }
  }
  return NO_PIECE;
}

void Position::update_castling_rights_for_move(Piece moving_piece,
                                               Side moving_side, Square from,
                                               Piece captured_piece,
                                               Square captured_square) {
  // King moved from its position
  if (moving_piece == KING) {
    if (moving_side == WHITE) {
      remove_castling_rights(WHITE_KINGSIDE);
      remove_castling_rights(WHITE_QUEENSIDE);
    } else {
      remove_castling_rights(BLACK_KINGSIDE);
      remove_castling_rights(BLACK_QUEENSIDE);
    }
  }

  // Rook moved from its position
  if (moving_piece == ROOK) {
    if (from == h1) {
      remove_castling_rights(WHITE_KINGSIDE);
    } else if (from == a1) {
      remove_castling_rights(WHITE_QUEENSIDE);
    } else if (from == h8) {
      remove_castling_rights(BLACK_KINGSIDE);
    } else if (from == a8) {
      remove_castling_rights(BLACK_QUEENSIDE);
    }
  }

  // Rook was captured on the position
  if (captured_piece == ROOK) {
    if (captured_square == h1) {
      remove_castling_rights(WHITE_KINGSIDE);
    } else if (captured_square == a1) {
      remove_castling_rights(WHITE_QUEENSIDE);
    } else if (captured_square == h8) {
      remove_castling_rights(BLACK_KINGSIDE);
    } else if (captured_square == a8) {
      remove_castling_rights(BLACK_QUEENSIDE);
    }
  }
}

Bitboard Position::get_piece(Side side, Piece piece) const {
  return pieces[side][piece];
}

Bitboard Position::get_occupancy(Side side) const {
  if (side == WHITE) {
    return white_occupancy;
  } else {
    return black_occupancy;
  }
}

Bitboard Position::get_all_occupancy() const { return all_occupancy; }

Side Position::get_side_to_move() const { return side_to_move; }

Bitboard Position::get_enimies(Side color) const {
  return color == WHITE ? black_occupancy : white_occupancy;
}

Square Position::get_en_passant_square() const { return en_passant_square; }

void Position::set_en_passant_square(Square square) {
  en_passant_square = square;
}

bool Position::has_castling_rights(CastlingRight right) const {
  return (castling_rights & right) != 0;
}

void Position::remove_castling_rights(CastlingRight right) {
  castling_rights &=
      static_cast<std::uint8_t>(~static_cast<std::uint8_t>(right));
}

void Position::clear_castling_rights() { castling_rights = 0; }

std::uint16_t Position::get_halfmove_clock() const { return halfmove_clock; }

std::uint16_t Position::get_fullmove_number() const { return fullmove_number; }

void Position::set_starting_position() {
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

  halfmove_clock = 0;
  fullmove_number = 1;

  side_to_move = Side::WHITE;
  castling_rights = WHITE_KINGSIDE | WHITE_QUEENSIDE | BLACK_KINGSIDE | BLACK_QUEENSIDE;
  update_occupancies();
  set_en_passant_square(NO_SQUARE);
}

void Position::print_position() const {
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

bool Position::is_in_check(Side side) {
  const auto king = get_piece(side, KING);
  const Square king_square = static_cast<Square>(std::countr_zero(king));
  const Side enemy = side == WHITE ? BLACK : WHITE;

  return is_square_attacked(*this, king_square, enemy);
}

/***
Check if position has insufficent material
 */

bool Position::has_insufficent_material() {
  // If there is a pawn, rook, or queen then not possible
  const Bitboard pawns = pieces[WHITE][PAWN] | pieces[BLACK][PAWN];
  const Bitboard rooks = pieces[WHITE][ROOK] | pieces[BLACK][ROOK];
  const Bitboard queens = pieces[WHITE][QUEEN] | pieces[BLACK][QUEEN];

  if (pawns != 0 || rooks != 0 || queens != 0) {
    return false;
  }

  // Now count number of each of the other pieces left
  const int white_knights = std::popcount(pieces[WHITE][KNIGHT]);
  const int black_knights = std::popcount(pieces[BLACK][KNIGHT]);
  const int white_bishops = std::popcount(pieces[WHITE][BISHOP]);
  const int black_bishops = std::popcount(pieces[BLACK][BISHOP]);

  const int total_knights = white_knights + black_knights;

  const int total_minor_pieces =
      white_knights + black_knights + white_bishops + black_bishops;

  // Only kings left, or king + 1 knight, king + 1 bishop

  if (total_minor_pieces == 0 || total_minor_pieces == 1) {
    return true;
  }
  // Case when only bishops of oppostie colors remain
  if (total_knights == 0) {
    Bitboard bishops = pieces[WHITE][BISHOP] | pieces[BLACK][BISHOP];
    bool has_light_square_bishop = false;
    bool has_dark_square_bishop = false;

    while (bishops != 0) {
      const int square_idx = std::countr_zero(bishops);
      // Remove lsb bishop
      bishops &= bishops - 1;

      // Get board position from square
      const int file = square_idx % 8;
      const int rank = square_idx / 8;

      // Light squares vs dark
      const bool is_light_square = ((file + rank) % 2);
      if (is_light_square) {
        has_light_square_bishop = true;
      } else {
        has_dark_square_bishop = true;
      }
    }

    return !(has_light_square_bishop && has_dark_square_bishop);
  }

  return false;
}

// Designing the Make move function to make moves on the board
bool Position::make_move(const Move &move) {
  const int from = static_cast<int>(move.from);
  const int to = static_cast<int>(move.to);

  // Validation
  if (from > 63 or from < 0 or to > 63 || to < 0) {
    return false;
  }

  const Side moving_side = get_side_to_move();
  const Side enemy_side = opposite_side(moving_side);
  Piece moving_piece = get_piece_on_square(move.from, moving_side); // Get Piece

  if (moving_piece == NO_PIECE) {
    return false;
  }
  // Check if square is a friendly peice
  if (get_bit(get_occupancy(moving_side), to)) {
    return false;
  }
  Piece capturing_piece = NO_PIECE;
  Square captured_square = NO_SQUARE;

  // Get the type of move it is for special move cases
  const bool normal_capture = move.type == MoveType::CAPTURE ||
                              move.type == MoveType::PROMOTION_CAPTURE;

  const bool promotion = move.type == MoveType::PROMOTION ||
                         move.type == MoveType::PROMOTION_CAPTURE;

  const bool en_passant = move.type == MoveType::EN_PASSANT;

  const bool kingside_castle = move.type == MoveType::KING_CASTLE;

  const bool queenside_castle = move.type == MoveType::QUEEN_CASTLE;

  // Getting the capturing piece if its a capture
  if (normal_capture) {
    capturing_piece = get_piece_on_square(move.to, enemy_side);
    if (capturing_piece == NO_PIECE) {
      return false;
    }

    captured_square = move.to;
  }
  // En Passant Case
  if (en_passant) {
    if (moving_piece != PAWN) {
      return false;
    }

    if (move.to != en_passant_square) {
      return false;
    }

    // En passant destination must be empty
    if (get_bit(all_occupancy, to)) {
      return false;
    }

    const int captured_index = moving_side == WHITE ? to - 8 : to + 8;

    if (captured_index < 0 || captured_index >= 64) {
      return false;
    }

    captured_square = static_cast<Square>(captured_index);

    capturing_piece = get_piece_on_square(captured_square, enemy_side);

    if (capturing_piece != PAWN) {
      return false;
    }
  }

  // Validation on promotion type move
  if (promotion) {
    if (moving_piece != PAWN) {
      return false;
    }

    if (move.promotion_piece != QUEEN && move.promotion_piece != ROOK &&
        move.promotion_piece != BISHOP && move.promotion_piece != KNIGHT) {
      return false;
    }
  }

  // Handling Castling Validation and setting rook movement squares
  Square rook_from = NO_SQUARE;
  Square rook_to = NO_SQUARE;
  if (kingside_castle) {
    if (moving_piece != KING) {
      return false;
    }
    if (moving_side == WHITE) {
      if (move.from != e1 || move.to != g1 ||
          !has_castling_rights(WHITE_KINGSIDE)) {
        return false;
      }

      rook_from = h1;
      rook_to = f1;
    } else {
      if (move.from != e8 || move.to != g8 ||
          !has_castling_rights(BLACK_KINGSIDE)) {
        return false;
      }

      rook_from = h8;
      rook_to = f8;
    }
  }

  if (queenside_castle) {
    if (moving_piece != KING) {
      return false;
    }

    if (moving_side == WHITE) {
      if (move.from != e1 || move.to != c1 ||
          !has_castling_rights(WHITE_QUEENSIDE)) {
        return false;
      }

      rook_from = a1;
      rook_to = d1;
    } else {
      if (move.from != e8 || move.to != c8 ||
          !has_castling_rights(BLACK_QUEENSIDE)) {
        return false;
      }

      rook_from = a8;
      rook_to = d8;
    }
  }
  // Update castling rights
  update_castling_rights_for_move(moving_piece, moving_side, move.from,
                                  capturing_piece, captured_square);

  // Remove piece from that position
  pop_bit(pieces[moving_side][moving_piece], move.from);

  if (capturing_piece != NO_PIECE) {
    pop_bit(pieces[enemy_side][capturing_piece], captured_square);
  }
  // If it is a promotion caputre have to set to new promotion piece
  if (promotion) {
    set_bit(pieces[moving_side][move.promotion_piece], to);
  } else {
    set_bit(pieces[moving_side][moving_piece], to);
  }

  // Movment of the rook on castling
  if (kingside_castle || queenside_castle) {
    pop_bit(pieces[moving_side][ROOK], static_cast<int>(rook_from));

    set_bit(pieces[moving_side][ROOK], static_cast<int>(rook_to));
  }
  // Update en passant square
  en_passant_square = NO_SQUARE;
  // If the double push then possible square for en passant
  if (move.type == MoveType::DOUBLE_PAWN_PUSH) {
    en_passant_square = static_cast<Square>((from + to) / 2);
  }

  // Update board occupanices
  update_occupancies();

  if (moving_piece == PAWN || capturing_piece != NO_PIECE) {
    halfmove_clock = 0;
  } else {
    ++halfmove_clock;
  }

  if (moving_side == BLACK) {
    ++fullmove_number;
  }

  // Change turn
  side_to_move = enemy_side;
  return true;
}

// Add this in for testing
bool Position::set_from_fen(const std::string &fen) {
  std::istringstream stream{fen};

  std::string board_field;
  std::string side_field;
  std::string castling_field;
  std::string en_passant_field;

  int parsed_halfmove_clock = 0;
  int parsed_fullmove_number = 1;

  // A complete FEN contains six fields.
  if (!(stream >> board_field >> side_field >> castling_field >>
        en_passant_field >> parsed_halfmove_clock >> parsed_fullmove_number)) {
    return false;
  }

  // Reject additional unexpected fields.
  std::string extra_field;
  if (stream >> extra_field) {
    return false;
  }

  if (parsed_halfmove_clock < 0 || parsed_fullmove_number < 1) {
    return false;
  }

  // Parse into a temporary Position so a failed parse does not corrupt
  // the current position.
  Position parsed{};

  parsed.pieces = {};
  parsed.white_occupancy = 0ULL;
  parsed.black_occupancy = 0ULL;
  parsed.all_occupancy = 0ULL;

  parsed.side_to_move = WHITE;
  parsed.en_passant_square = NO_SQUARE;
  parsed.castling_rights = 0;

  /*
   * Parse the board.
   *
   * FEN begins at rank 8 and moves toward rank 1.
   */
  int rank = 7;
  int file = 0;

  for (const char symbol : board_field) {
    if (symbol == '/') {
      // Every completed rank must contain exactly eight squares.
      if (file != 8 || rank == 0) {
        return false;
      }

      --rank;
      file = 0;
      continue;
    }

    // Digits represent consecutive empty squares.
    if (symbol >= '1' && symbol <= '8') {
      file += symbol - '0';

      if (file > 8) {
        return false;
      }

      continue;
    }

    if (file >= 8) {
      return false;
    }

    Side piece_side;
    Piece piece;

    switch (symbol) {
    case 'P':
      piece_side = WHITE;
      piece = PAWN;
      break;

    case 'N':
      piece_side = WHITE;
      piece = KNIGHT;
      break;

    case 'B':
      piece_side = WHITE;
      piece = BISHOP;
      break;

    case 'R':
      piece_side = WHITE;
      piece = ROOK;
      break;

    case 'Q':
      piece_side = WHITE;
      piece = QUEEN;
      break;

    case 'K':
      piece_side = WHITE;
      piece = KING;
      break;

    case 'p':
      piece_side = BLACK;
      piece = PAWN;
      break;

    case 'n':
      piece_side = BLACK;
      piece = KNIGHT;
      break;

    case 'b':
      piece_side = BLACK;
      piece = BISHOP;
      break;

    case 'r':
      piece_side = BLACK;
      piece = ROOK;
      break;

    case 'q':
      piece_side = BLACK;
      piece = QUEEN;
      break;

    case 'k':
      piece_side = BLACK;
      piece = KING;
      break;

    default:
      return false;
    }

    const int square = rank * 8 + file;

    set_bit(parsed.pieces[piece_side][piece], square);

    ++file;
  }

  // The parser must finish at the end of rank 1.
  if (rank != 0 || file != 8) {
    return false;
  }

  /*
   * Parse side to move.
   */
  if (side_field == "w") {
    parsed.side_to_move = WHITE;
  } else if (side_field == "b") {
    parsed.side_to_move = BLACK;
  } else {
    return false;
  }

  /*
   * Parse castling rights.
   */
  if (castling_field != "-") {
    for (const char right : castling_field) {
      switch (right) {
      case 'K':
        parsed.castling_rights |= static_cast<std::uint8_t>(WHITE_KINGSIDE);
        break;

      case 'Q':
        parsed.castling_rights |= static_cast<std::uint8_t>(WHITE_QUEENSIDE);
        break;

      case 'k':
        parsed.castling_rights |= static_cast<std::uint8_t>(BLACK_KINGSIDE);
        break;

      case 'q':
        parsed.castling_rights |= static_cast<std::uint8_t>(BLACK_QUEENSIDE);
        break;

      default:
        return false;
      }
    }
  }

  /*
   * Parse en passant target square.
   */
  if (en_passant_field != "-") {
    if (en_passant_field.size() != 2) {
      return false;
    }

    const char ep_file = en_passant_field[0];
    const char ep_rank = en_passant_field[1];

    if (ep_file < 'a' || ep_file > 'h') {
      return false;
    }

    // A FEN en passant target can only be on rank 3 or rank 6.
    if (ep_rank != '3' && ep_rank != '6') {
      return false;
    }

    const int file_index = ep_file - 'a';
    const int rank_index = ep_rank - '1';
    const int square_index = rank_index * 8 + file_index;

    parsed.en_passant_square = static_cast<Square>(square_index);
  }

  /*
   * Store these if Position has the corresponding members.
   */
  parsed.halfmove_clock = parsed_halfmove_clock;
  parsed.fullmove_number = parsed_fullmove_number;

  parsed.update_occupancies();

  /*
   * Ensure each side has exactly one king.
   */
  if (std::popcount(parsed.pieces[WHITE][KING]) != 1 ||
      std::popcount(parsed.pieces[BLACK][KING]) != 1) {
    return false;
  }

  // Parsing succeeded, so replace the current position
  *this = parsed;

  return true;
}