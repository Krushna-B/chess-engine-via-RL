#pragma once

#include "board.hpp"

namespace gui {

// Layout (shared by the renderer and the controller's mouse hit-testing so the
// board origin is defined in exactly one place).
inline constexpr int SQUARE_SIZE = 90;
inline constexpr int BOARD_SIZE = SQUARE_SIZE * 8;
inline constexpr int TOP_BAR = 56;    // status strip above the board
inline constexpr int BOTTOM_BAR = 44; // controls hint below the board
inline constexpr int BOARD_X = 0;
inline constexpr int BOARD_Y = TOP_BAR;
inline constexpr int WINDOW_WIDTH = BOARD_SIZE;
inline constexpr int WINDOW_HEIGHT = TOP_BAR + BOARD_SIZE + BOTTOM_BAR;

// Piece textures need a GL context, so these bracket the render loop:
void load_assets();   // after InitWindow
void unload_assets(); // before CloseWindow

void draw(const Position &position);

} // namespace gui
