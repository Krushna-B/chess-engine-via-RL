#include "position_encoder.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace {

constexpr std::size_t HISTORY_LENGTH = 8;
constexpr std::size_t PLANES_PER_POSITION = 13;

std::size_t encoded_square(int square, Side side_to_move) {
  if (side_to_move == WHITE) {
    return static_cast<std::size_t>(square);
  }

  return static_cast<std::size_t>(square ^ 56);
}

void fill_plane(EncodedPositionHistory &encoded, std::size_t plane,
                float value) {
  auto first = encoded.begin() +
               static_cast<std::ptrdiff_t>(plane * POSITION_PLANE_SIZE);
  std::fill_n(first, POSITION_PLANE_SIZE, value);
}

bool has_previous_occurrence(const PositionHistory &history,
                             std::size_t position_index) {
  u64 position_hash = history.get_position(position_index).hash();

  for (std::size_t i = 0; i < position_index; ++i) {
    if (history.get_position(i).hash() == position_hash) {
      return true;
    }
  }

  return false;
}

} // namespace

EncodedPositionHistory encode_position_history(const PositionHistory &history) {
  if (history.size() == 0) {
    throw std::invalid_argument("Cannot encode empty position history");
  }

  EncodedPositionHistory encoded{};
  Side current_side =
      history.get_position(history.size() - 1).get_side_to_move();
  Side opponent_side = current_side == WHITE ? BLACK : WHITE;

  constexpr std::array<Piece, 6> PIECE_TYPES{PAWN, KNIGHT, BISHOP,
                                             ROOK, QUEEN,  KING};

  std::size_t history_length = std::min(HISTORY_LENGTH, history.size());

  for (std::size_t history_offset = 0; history_offset < history_length;
       ++history_offset) {
    std::size_t position_index = history.size() - 1 - history_offset;
    const Position &position = history.get_position(position_index);
    std::size_t base_plane = history_offset * PLANES_PER_POSITION;

    for (std::size_t piece_index = 0; piece_index < PIECE_TYPES.size();
         ++piece_index) {
      for (int square = 0; square < 64; ++square) {
        Bitboard square_mask = Bitboard{1} << square;
        std::size_t target_square = encoded_square(square, current_side);

        if ((position.get_piece(current_side, PIECE_TYPES[piece_index]) &
             square_mask) != 0) {
          encoded[(base_plane + piece_index) * POSITION_PLANE_SIZE +
                  target_square] = 1.0f;
        }

        if ((position.get_piece(opponent_side, PIECE_TYPES[piece_index]) &
             square_mask) != 0) {
          encoded[(base_plane + 6 + piece_index) * POSITION_PLANE_SIZE +
                  target_square] = 1.0f;
        }
      }
    }

    if (has_previous_occurrence(history, position_index)) {
      fill_plane(encoded, base_plane + 12, 1.0f);
    }
  }

  const Position &position = history.get_position(history.size() - 1);
  CastlingRight own_queenside =
      current_side == WHITE ? WHITE_QUEENSIDE : BLACK_QUEENSIDE;
  CastlingRight own_kingside =
      current_side == WHITE ? WHITE_KINGSIDE : BLACK_KINGSIDE;
  CastlingRight opponent_queenside =
      current_side == WHITE ? BLACK_QUEENSIDE : WHITE_QUEENSIDE;
  CastlingRight opponent_kingside =
      current_side == WHITE ? BLACK_KINGSIDE : WHITE_KINGSIDE;

  if (position.has_castling_rights(own_queenside)) {
    fill_plane(encoded, 104, 1.0f);
  }
  if (position.has_castling_rights(own_kingside)) {
    fill_plane(encoded, 105, 1.0f);
  }
  if (position.has_castling_rights(opponent_queenside)) {
    fill_plane(encoded, 106, 1.0f);
  }
  if (position.has_castling_rights(opponent_kingside)) {
    fill_plane(encoded, 107, 1.0f);
  }
  if (current_side == BLACK) {
    fill_plane(encoded, 108, 1.0f);
  }

  fill_plane(encoded, 109,
             static_cast<float>(position.get_halfmove_clock()) / 99.0f);
  fill_plane(encoded, 111, 1.0f);

  return encoded;
}
