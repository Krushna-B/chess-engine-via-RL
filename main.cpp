#include "board.hpp"
#include "gui.hpp"
#include "move_generator.hpp"
#include "move_list.hpp"
#include "raylib.h"

// raylib defines WHITE/BLACK as Color macros, which collide with
// Side::WHITE/BLACK.
#undef WHITE
#undef BLACK

void run_chess_app(Position position);

int main() {
  Position position{};
  position.set_starting_position();
  position.print_position();

  MoveList moves{};

  generate_all_pseudo_moves(moves, position);
  moves.print();
  std::cout << "Number of possible moves is: " << moves.get_count();

  // run_chess_app(position);

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