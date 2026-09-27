#include "game.hpp"

#include <array>
#include <iostream>

int main() {
  Game game{};

  const std::array<Move, 8> moves = {
      Move{g1, f3}, Move{g8, f6}, Move{f3, g1}, Move{f6, g8},
      Move{g1, f3}, Move{g8, f6}, Move{f3, g1}, Move{f6, g8},
  };

  for (std::size_t i = 0; i < moves.size(); ++i) {
    if (!game.play_move(moves[i])) {
      std::cerr << "Move " << i << " was rejected\n";
      return 1;
    }

    if (i < moves.size() - 1 && game.get_status() != GameStatus::ONGOING) {
      std::cerr << "Game ended before the third repetition\n";
      return 1;
    }
  }

  if (game.get_status() != GameStatus::DRAW_THREE_FOLD_REPITITION) {
    std::cerr << "Game did not end by threefold repetition\n";
    return 1;
  }

  return 0;
}
