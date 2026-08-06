#include "board.hpp"
#include "game.hpp"
#include "gui.hpp"
#include "move_list.hpp"
#include "random_engine.hpp"
#include "raylib.h"
#include <ostream>

// raylib defines WHITE/BLACK as Color macros, which collide with
// Side::WHITE/BLACK.
#undef WHITE
#undef BLACK

void run_chess_app(Position position);

int main() {
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
  return 0;
}

void run_chess_app(Position position) {
  constexpr int window_width = 720;
  constexpr int window_height = 720;

  InitWindow(window_width, window_height, "Chess Engine");
  SetTargetFPS(60);
  while (!WindowShouldClose()) {
    BeginDrawing();

    gui::draw(position);

    EndDrawing();
  }

  CloseWindow();
}