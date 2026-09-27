#include "mcts.hpp"

#include <array>
#include <iostream>
#include <memory>

int main() {
  Position position{};
  position.set_starting_position();

  auto root = std::make_unique<Node>(position);

  const std::array<Move, 8> moves = {
      Move{g1, f3}, Move{g8, f6}, Move{f3, g1}, Move{f6, g8},
      Move{g1, f3}, Move{g8, f6}, Move{f3, g1}, Move{f6, g8},
  };

  for (std::size_t i = 0; i < moves.size(); ++i) {
    Position child_position = root->state;

    if (!child_position.make_move(moves[i])) {
      std::cerr << "Move " << i << " was rejected\n";
      return 1;
    }

    root->children.push_back(
        std::make_unique<Node>(child_position, moves[i], root.get(), 1.0f));
    root = advance_root(std::move(root), 0);

    if (i < moves.size() - 1 && root->is_threefold_repetition()) {
      std::cerr << "MCTS detected repetition too early\n";
      return 1;
    }
  }

  if (!root->is_threefold_repetition()) {
    std::cerr << "MCTS did not detect threefold repetition\n";
    return 1;
  }

  if (!is_terminal(*root) || terminal_value(*root) != 0.0f) {
    std::cerr << "MCTS repetition was not a terminal draw\n";
    return 1;
  }

  Node restored_root{root->state, root->position_history};

  if (!restored_root.is_threefold_repetition()) {
    std::cerr << "Restored MCTS root lost its repetition history\n";
    return 1;
  }

  return 0;
}
