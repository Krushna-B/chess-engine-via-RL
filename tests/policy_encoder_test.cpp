#include "move_generator.hpp"
#include "move_list.hpp"
#include "policy_encoder.hpp"

#include <iostream>
#include <stdexcept>

namespace {

bool check(bool condition, const char *message) {
  if (!condition) {
    std::cerr << message << '\n';
    return false;
  }

  return true;
}

bool check_legal_moves(Position &position) {
  MoveList moves{};
  generate_legal_moves(moves, position);

  for (const Move &move : moves.get_moves()) {
    std::size_t index = encode_lc0_policy_move(position, move);
    Move decoded = decode_lc0_policy_move(position, index);

    if (!(decoded == move)) {
      std::cerr << "Policy round trip failed for move " << move.from << move.to
                << '\n';
      return false;
    }
  }

  return true;
}

} // namespace

int main() {
  Position starting_position{};
  starting_position.set_starting_position();

  if (!check(encode_lc0_policy_move(starting_position, Move{a1, b1}) == 0,
             "a1b1 did not map to index zero") ||
      !check(encode_lc0_policy_move(starting_position, Move{a1, h8}) == 22,
             "a1h8 did not map to index 22") ||
      !check(encode_lc0_policy_move(starting_position, Move{b1, a1}) == 23,
             "b1a1 did not map to index 23") ||
      !check(encode_lc0_policy_move(starting_position, Move{h8, g8}) == 1791,
             "Last regular move did not map to index 1791") ||
      !check_legal_moves(starting_position)) {
    return 1;
  }

  Position white_promotion{};
  if (!white_promotion.set_from_fen(
          "4k3/P7/8/8/8/8/8/4K3 w - - 0 1")) {
    std::cerr << "Could not create White promotion position\n";
    return 1;
  }

  Move knight_promotion{a7, a8, MoveType::PROMOTION, KNIGHT};
  Move queen_promotion{a7, a8, MoveType::PROMOTION, QUEEN};
  Move rook_promotion{a7, a8, MoveType::PROMOTION, ROOK};
  Move bishop_promotion{a7, a8, MoveType::PROMOTION, BISHOP};

  if (!check(encode_lc0_policy_move(white_promotion, queen_promotion) == 1792,
             "Queen promotion did not map to index 1792") ||
      !check(encode_lc0_policy_move(white_promotion, rook_promotion) == 1793,
             "Rook promotion did not map to index 1793") ||
      !check(encode_lc0_policy_move(white_promotion, bishop_promotion) == 1794,
             "Bishop promotion did not map to index 1794") ||
      !check(encode_lc0_policy_move(white_promotion, knight_promotion) < 1792,
             "Knight promotion did not use a regular move index") ||
      !check_legal_moves(white_promotion)) {
    return 1;
  }

  Position black_position{};
  if (!black_position.set_from_fen(
          "4k3/4p3/8/8/8/8/8/4K3 b - - 0 1")) {
    std::cerr << "Could not create Black policy position\n";
    return 1;
  }

  Position white_position{};
  if (!white_position.set_from_fen(
          "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")) {
    std::cerr << "Could not create White policy position\n";
    return 1;
  }

  Move black_push{e7, e5, MoveType::DOUBLE_PAWN_PUSH};
  Move white_push{e2, e4, MoveType::DOUBLE_PAWN_PUSH};

  if (!check(encode_lc0_policy_move(black_position, black_push) ==
                 encode_lc0_policy_move(white_position, white_push),
             "Black move did not use rank-flipped policy index") ||
      !check(decode_lc0_policy_move(
                 black_position,
                 encode_lc0_policy_move(black_position, black_push)) ==
                 black_push,
             "Black move did not round trip") ||
      !check_legal_moves(black_position)) {
    return 1;
  }

  Position castling_position{};
  if (!castling_position.set_from_fen(
          "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")) {
    std::cerr << "Could not create castling position\n";
    return 1;
  }

  if (!check_legal_moves(castling_position)) {
    return 1;
  }

  try {
    encode_lc0_policy_move(starting_position, Move{a1, b4});
    std::cerr << "Invalid move was accepted by policy encoder\n";
    return 1;
  } catch (const std::invalid_argument &) {
  }

  try {
    decode_lc0_policy_move(starting_position, LC0_POLICY_SIZE);
    std::cerr << "Invalid policy index was accepted\n";
    return 1;
  } catch (const std::out_of_range &) {
  }

  return 0;
}
