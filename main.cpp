#include "board.hpp"
#include "gui.hpp"
#include "raylib.h"

void run_chess_app(Position position);

int main() {
  Position position{};
  position.set_starting_position();

  run_chess_app(position);

  return 0;
}

void run_chess_app(Position position) {
  constexpr int window_width = 720;
  constexpr int window_height = 720;

  InitWindow(window_width, window_height, "Chess Engine");
  SetTargetFPS(60);
  while (!WindowShouldClose()) {
    BeginDrawing();

    ClearBackground(DARKGRAY);

    gui::draw(position);

    EndDrawing();
  }

  CloseWindow();
}