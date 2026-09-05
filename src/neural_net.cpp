#include "neural_net.hpp"
#include "board.hpp"
#include "move_generator.hpp"
#include "move_list.hpp"
#include <array>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

u64 encode_underpromotion_piece(const Move &move, int del_rank, int del_file);
u64 encode_knight_type(int rank_change, int file_change);
int find_direction(int rank_change, int file_change);

int canonicalize_square(int square, Side side_to_move);

/**
Turns a move and the square from [73, 64] to the linear action index in
[0, 4671].
*/
constexpr u64 policy_idx(u64 move_type, u64 original_square) {
  return move_type * BOARD_SIZE + original_square;
}

/**
Takes a move generator from our mcts and then turns into index action index
which is the move matching the neural net's ouput in order to get its logit
*/
u64 encode_move(const Position &position, const Move &move) {
  int physical_from = static_cast<int>(move.from);
  int physical_to = static_cast<int>(move.to);

  Side side_to_move = position.get_side_to_move();

  int from = canonicalize_square(physical_from, side_to_move);
  int to = canonicalize_square(physical_to, side_to_move);

  int from_rank = from / 8;
  int from_file = from % 8;

  int to_rank = to / 8;
  int to_file = to % 8;

  int rank_change = to_rank - from_rank;
  int file_change = to_file - from_file;

  u64 move_type{};

  bool underpromotion =
      move.promotion_piece != QUEEN && move.promotion_piece != NO_PIECE;

  if (underpromotion) {
    move_type = encode_underpromotion_piece(move, rank_change, file_change);
  } else {
    bool knight_movement =
        (std::abs(rank_change) == 2 && std::abs(file_change) == 1) ||
        (std::abs(rank_change) == 1 && std::abs(file_change) == 2);

    if (knight_movement) {
      move_type = encode_knight_type(rank_change, file_change);
    } else {
      bool straight = rank_change == 0 || file_change == 0;
      bool diagonal = std::abs(rank_change) == std::abs(file_change);

      if (!straight && !diagonal) {
        throw std::invalid_argument("Move is not straight or diagonal");
      }

      int distance = std::max(std::abs(rank_change), std::abs(file_change));

      int direction = find_direction(rank_change, file_change);
      move_type = static_cast<u64>(direction * 7 + distance - 1);
    }
  }
  u64 action = move_type * 64 + static_cast<u64>(from);
  return action;
}

void validate_move_encoding(Position &position) {
  MoveList moves{};
  generate_legal_moves(moves, position);
  auto legal_moves = moves.get_moves();
  std::array<bool, POLICY_SIZE> used{};
  std::cout << "Legal Moves " << legal_moves.size() << "\n";
  for (const Move &move : legal_moves) {
    u64 action = encode_move(position, move);
    if (action > POLICY_SIZE) {
      throw std::runtime_error("Encoded action exceeds the policy array size");
    }
    if (used[action]) {
      std::cerr << "Collsion for move from " << static_cast<int>(move.from)
                << " to " << static_cast<int>(move.to) << " at action "
                << action << "\n";
      throw std::runtime_error("Two legal moves have same action idx");
    }
    used[action] = true;
  }
  std::cout << "Move encoding passed" << "\n";
}

u64 encode_underpromotion_piece(const Move &move, int del_rank, int del_file) {
  u64 piece_group;

  switch (move.promotion_piece) {
  case KNIGHT:
    piece_group = 0;
    break;
  case BISHOP:
    piece_group = 1;
    break;
  case ROOK:
    piece_group = 2;
    break;
  default:
    throw std::invalid_argument(
        "Expected underpromotion piece to be knight, bishop, rook");
  }
  // file_change either -1,0,1 -> 0,1,2
  u64 direction = static_cast<u64>(del_file + 1);

  return 64 + piece_group * 3 + direction;
}

u64 encode_knight_type(int rank_change, int file_change) {
  constexpr std::array<std::pair<int, int>, 8> KNIGHT_MOVES = {
      {{2, 1}, {1, 2}, {-1, 2}, {-2, 1}, {-2, -1}, {-1, -2}, {1, -2}, {2, -1}}};
  for (u64 i{}; i < KNIGHT_MOVES.size(); i++) {
    if (KNIGHT_MOVES[i].first == rank_change &&
        KNIGHT_MOVES[i].second == file_change) {
      return 56 + i;
    }
  }
  throw std::invalid_argument("Invalid knight movement");
}
int find_direction(int rank_change, int file_change) {
  if ((rank_change == 0 && file_change == 0) ||
      (rank_change != 0 && file_change != 0 &&
       std::abs(rank_change) != std::abs(file_change)))
    throw std::invalid_argument(
        "find_direction: not a straight line or diagonal");

  int r = (rank_change > 0) - (rank_change < 0);
  int f = (file_change > 0) - (file_change < 0);

  static const int table[3][3] = {
      /* f=-1  f=0  f=+1 */
      {5, 4, 3},  /* r = -1 */
      {6, -1, 2}, /* r =  0 */
      {7, 0, 1}   /* r = +1 */
  };

  return table[r + 1][f + 1];
}

int canonicalize_square(int square, Side side_to_move) {
  if (side_to_move == Side::WHITE) {
    return square;
  }
  return 63 - square;
}

EncodedPosition encode_position(const Position &position) {
  EncodedPosition encoded{};
  encoded.fill(0.0f);

  Side current_side = position.get_side_to_move();

  Side opponent_side = current_side == Side::WHITE ? Side::BLACK : Side::WHITE;

  constexpr std::array<Piece, 6> PIECE_TYPES{PAWN, KNIGHT, BISHOP,
                                             ROOK, QUEEN,  KING};

  /*
   * Features 0-5:  current-player pieces
   * Features 6-11: opponent pieces
   */
  for (std::size_t piece_index = 0; piece_index < PIECE_TYPES.size();
       ++piece_index) {
    Piece piece_type = PIECE_TYPES[piece_index];

    Bitboard own_pieces = position.get_piece(current_side, piece_type);

    Bitboard opponent_pieces = position.get_piece(opponent_side, piece_type);

    for (int physical_square = 0; physical_square < 64; ++physical_square) {
      Bitboard square_mask = Bitboard{1} << physical_square;

      int canonical_square = canonicalize_square(physical_square, current_side);

      std::size_t square = static_cast<std::size_t>(canonical_square);

      if ((own_pieces & square_mask) != 0) {
        std::size_t feature = piece_index;

        std::size_t index = square * SQUARE_FEATURES + feature;

        encoded[index] = 1.0f;
      }

      if ((opponent_pieces & square_mask) != 0) {
        std::size_t feature = 6 + piece_index;

        std::size_t index = square * SQUARE_FEATURES + feature;

        encoded[index] = 1.0f;
      }
    }
  }

  /*
   * Determine castling rights relative to the
   * current player.
   *
   * Replace these four CastlingRight names with
   * the names from your enum.
   */
  bool white_kingside =
      position.has_castling_rights(CastlingRight::WHITE_KINGSIDE);

  bool white_queenside =
      position.has_castling_rights(CastlingRight::WHITE_QUEENSIDE);

  bool black_kingside =
      position.has_castling_rights(CastlingRight::BLACK_KINGSIDE);

  bool black_queenside =
      position.has_castling_rights(CastlingRight::BLACK_QUEENSIDE);

  bool own_kingside;
  bool own_queenside;
  bool opponent_kingside;
  bool opponent_queenside;

  if (current_side == Side::WHITE) {
    own_kingside = white_kingside;
    own_queenside = white_queenside;

    opponent_kingside = black_kingside;
    opponent_queenside = black_queenside;
  } else {
    own_kingside = black_kingside;
    own_queenside = black_queenside;

    opponent_kingside = white_kingside;
    opponent_queenside = white_queenside;
  }

  float normalized_halfmove = std::min(
      static_cast<float>(position.get_halfmove_clock()) / 100.0f, 1.0f);

  /*
   * Features 12-15 and 17 describe the whole
   * position, so repeat them for all 64 tokens.
   */
  for (std::size_t square = 0; square < BOARD_SIZE; ++square) {
    std::size_t base = square * SQUARE_FEATURES;

    encoded[base + 12] = own_kingside ? 1.0f : 0.0f;

    encoded[base + 13] = own_queenside ? 1.0f : 0.0f;

    encoded[base + 14] = opponent_kingside ? 1.0f : 0.0f;

    encoded[base + 15] = opponent_queenside ? 1.0f : 0.0f;

    encoded[base + 17] = normalized_halfmove;
  }

  /*
   * Feature 16 identifies the canonical
   * en-passant target square.
   */
  Square en_passant = position.get_en_passant_square();

  // Replace NO_SQUARE with your sentinel value.
  if (en_passant != NO_SQUARE) {
    int physical_square = static_cast<int>(en_passant);

    int canonical_square = canonicalize_square(physical_square, current_side);

    std::size_t index =
        static_cast<std::size_t>(canonical_square) * SQUARE_FEATURES + 16;

    encoded[index] = 1.0f;
  }

  return encoded;
}