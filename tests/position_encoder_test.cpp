#include "position_encoder.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

float value(const EncodedPositionHistory &encoded, std::size_t plane,
            Square square) {
  return encoded[plane * POSITION_PLANE_SIZE +
                 static_cast<std::size_t>(square)];
}

bool check(bool condition, const char *message) {
  if (!condition) {
    std::cerr << message << '\n';
    return false;
  }

  return true;
}

bool check_plane(const EncodedPositionHistory &encoded, std::size_t plane,
                 float expected, const char *message) {
  for (std::size_t square = 0; square < POSITION_PLANE_SIZE; ++square) {
    float actual = encoded[plane * POSITION_PLANE_SIZE + square];

    if (std::abs(actual - expected) > 0.000001f) {
      std::cerr << message << '\n';
      return false;
    }
  }

  return true;
}

} // namespace

int main() {
  Position starting_position{};
  starting_position.set_starting_position();

  PositionHistory starting_history{};
  starting_history.add_position(starting_position);
  EncodedPositionHistory starting =
      encode_position_history(starting_history);

  if (!check(value(starting, 0, e2) == 1.0f,
             "White pawn was not encoded on e2") ||
      !check(value(starting, 5, e1) == 1.0f,
             "White king was not encoded on e1") ||
      !check(value(starting, 6, e7) == 1.0f,
             "Black pawn was not encoded on e7") ||
      !check(value(starting, 11, e8) == 1.0f,
             "Black king was not encoded on e8") ||
      !check_plane(starting, 13, 0.0f,
                   "Unavailable history was not zero-filled") ||
      !check_plane(starting, 104, 1.0f,
                   "Own queenside castling plane was not set") ||
      !check_plane(starting, 105, 1.0f,
                   "Own kingside castling plane was not set") ||
      !check_plane(starting, 106, 1.0f,
                   "Opponent queenside castling plane was not set") ||
      !check_plane(starting, 107, 1.0f,
                   "Opponent kingside castling plane was not set") ||
      !check_plane(starting, 108, 0.0f,
                   "White side-to-move plane was not zero") ||
      !check_plane(starting, 110, 0.0f,
                   "Constant zero plane was not zero") ||
      !check_plane(starting, 111, 1.0f,
                   "Constant one plane was not set")) {
    return 1;
  }

  Position black_position{};
  if (!black_position.set_from_fen(
          "4k3/8/8/8/8/8/P7/4K3 b qK - 37 1")) {
    std::cerr << "Could not create Black test position\n";
    return 1;
  }

  PositionHistory black_history{};
  black_history.add_position(black_position);
  EncodedPositionHistory black = encode_position_history(black_history);

  if (!check(value(black, 5, e1) == 1.0f,
             "Black king was not rank-flipped to e1") ||
      !check(value(black, 6, a7) == 1.0f,
             "White pawn was not rank-flipped to a7") ||
      !check(value(black, 11, e8) == 1.0f,
             "White king was not rank-flipped to e8") ||
      !check_plane(black, 104, 1.0f,
                   "Black queenside castling plane was not set") ||
      !check_plane(black, 105, 0.0f,
                   "Black kingside castling plane was set") ||
      !check_plane(black, 106, 0.0f,
                   "White queenside castling plane was set") ||
      !check_plane(black, 107, 1.0f,
                   "White kingside castling plane was not set") ||
      !check_plane(black, 108, 1.0f,
                   "Black side-to-move plane was not set") ||
      !check(std::abs(value(black, 109, a1) - 37.0f / 99.0f) <
                 0.000001f,
             "Rule-50 plane was not normalized")) {
    return 1;
  }

  Position different_position{};
  if (!different_position.set_from_fen(
          "4k3/8/8/8/8/8/8/4K3 b - - 0 1")) {
    std::cerr << "Could not create history test position\n";
    return 1;
  }

  Position repeated_position{};
  if (!repeated_position.set_from_fen(
          "4k3/8/8/8/8/8/8/4K3 w - - 0 1")) {
    std::cerr << "Could not create repetition test position\n";
    return 1;
  }

  PositionHistory repeated_history{};
  repeated_history.add_position(repeated_position);
  repeated_history.add_position(different_position);
  repeated_history.add_position(repeated_position);
  EncodedPositionHistory repeated =
      encode_position_history(repeated_history);

  if (!check_plane(repeated, 12, 1.0f,
                   "Repeated current position plane was not set") ||
      !check_plane(repeated, 25, 0.0f,
                   "Different historical repetition plane was set") ||
      !check_plane(repeated, 38, 0.0f,
                   "First occurrence repetition plane was set") ||
      !check(value(repeated, 5, e1) == 1.0f,
             "Current history position was not in slot zero") ||
      !check(value(repeated, 18, e1) == 1.0f,
             "Previous history position was not in slot one") ||
      !check(value(repeated, 31, e1) == 1.0f,
             "Oldest history position was not in slot two")) {
    return 1;
  }

  try {
    PositionHistory empty_history{};
    encode_position_history(empty_history);
    std::cerr << "Empty history did not throw\n";
    return 1;
  } catch (const std::invalid_argument &) {
  }

  return 0;
}
