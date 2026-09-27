#include "move_list.hpp"
#include "position_history.hpp"

#include <iostream>

bool check(bool condition, const char *message) {
  if (!condition) {
    std::cerr << message << '\n';
  }

  return condition;
}

int main() {
  PositionHistory history{};
  Position starting_position{};
  starting_position.set_starting_position();

  if (!check(!history.is_threefold_repetition(),
             "Empty history reported a repetition")) {
    return 1;
  }

  history.add_position(starting_position);
  history.add_position(starting_position);
  history.add_position(starting_position);

  if (!check(history.is_threefold_repetition(),
             "Three matching positions were not detected")) {
    return 1;
  }

  Position different_position = starting_position;

  if (!different_position.make_move(Move{e2, e4, MoveType::DOUBLE_PAWN_PUSH})) {
    std::cerr << "Could not create a different position\n";
    return 1;
  }

  history.add_position(different_position);

  if (!check(!history.is_threefold_repetition(),
             "A different current position reported a repetition")) {
    return 1;
  }

  history.add_position(different_position);
  history.add_position(different_position);

  if (!check(history.is_threefold_repetition(),
             "Repeated current position was not detected")) {
    return 1;
  }

  if (!check(history.size() == 6,
             "Position history did not retain every position")) {
    return 1;
  }

  if (!check(history.get_position(3).hash() == different_position.hash(),
             "Stored position did not match the original position")) {
    return 1;
  }

  history.clear();

  if (!check(!history.is_threefold_repetition(),
             "Cleared history reported a repetition")) {
    return 1;
  }

  if (!check(history.size() == 0, "Cleared history retained positions")) {
    return 1;
  }

  return 0;
}
