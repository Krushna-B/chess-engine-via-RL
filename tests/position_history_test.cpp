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

  if (!check(!history.is_threefold_repetition(),
             "Empty history reported a repetition")) {
    return 1;
  }

  history.add_position(10);
  history.add_position(20);
  history.add_position(10);
  history.add_position(20);
  history.add_position(10);

  if (!check(history.is_threefold_repetition(),
             "Three matching positions were not detected")) {
    return 1;
  }

  history.add_position(30);

  if (!check(!history.is_threefold_repetition(),
             "A different current position reported a repetition")) {
    return 1;
  }

  history.add_position(30);
  history.add_position(30);

  if (!check(history.is_threefold_repetition(),
             "Repeated current position was not detected")) {
    return 1;
  }

  history.clear();

  if (!check(!history.is_threefold_repetition(),
             "Cleared history reported a repetition")) {
    return 1;
  }

  return 0;
}
