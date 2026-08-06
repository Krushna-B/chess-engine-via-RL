#include "board.hpp"
#include "game.hpp"
#include "game_controller.hpp"

#include "move_list.hpp"
#include "random_engine.hpp"
#include <ostream>

// raylib defines WHITE/BLACK as Color macros, which collide with
// Side::WHITE/BLACK.
#undef WHITE
#undef BLACK

const Side USER_SIDE = Side::WHITE;
int run_chess_app(Side users_side);

int main() { return run_chess_app(USER_SIDE); }
int run_chess_app(Side users_side) {
  GameController game_controller(users_side);
  game_controller.run();
  return 0;
}

int engines_play_against_each_other() {
  Game game{};
  int move_count{};
  constexpr auto max_moves = 500;

  // Create engines
  RandomEngine white_engine{};
  RandomEngine black_engine{};

  while (game.get_status() == GameStatus::ONGOING && move_count < max_moves) {
    const Position &position = game.get_position();

    RandomEngine &engine = position.get_side_to_move() == Side::WHITE
                               ? white_engine
                               : black_engine;

    // Engine chooses a move

    std::optional<Move> move = engine.choose_move(position);

    if (!game.play_move(*move)) {
      std::cerr << "Engine produced an illegal move\n";
      return 1;
    }
    ++move_count;

    game.get_position().print_position();
    std::cout << game.get_status() << std::endl;
    std::cout << '\n';
  }

  std::cout << "Game finished after " << move_count
            << " moves: " << "result is " << game.get_status() << std::endl;
}