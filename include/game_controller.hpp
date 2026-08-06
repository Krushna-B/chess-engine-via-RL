#pragma once

#include "game.hpp"
#include "gui.hpp"
#include "move_list.hpp"
#include "random_engine.hpp"

#include <optional>
#include <vector>

#include <raylib.h>

// raylib defines WHITE and BLACK as macros.
#undef WHITE
#undef BLACK

class GameController {
public:
  explicit GameController(Side human_side);

  void run();

private:
  static constexpr int WINDOW_WIDTH = gui::WINDOW_WIDTH;
  static constexpr int WINDOW_HEIGHT = gui::WINDOW_HEIGHT;

  static constexpr double ENGINE_MOVE_DELAY = 0.35;

  Game game{};
  RandomEngine engine{};
  Font ui_font{};

  Side human_side;
  Side engine_side;

  Rectangle board_bounds{
      static_cast<float>(gui::BOARD_X), static_cast<float>(gui::BOARD_Y),
      static_cast<float>(gui::BOARD_SIZE), static_cast<float>(gui::BOARD_SIZE)};

  MoveList legal_moves{};

  std::optional<Square> selected_square{};
  std::vector<Move> selected_moves{};

  double engine_turn_start{-1.0};
  bool controller_error{false};

  void update();
  void draw();

  void start_new_game(Side new_human_side);
  void refresh_legal_moves();

  void handle_human_turn();
  void handle_engine_turn();

  void select_square(Square square);
  void clear_selection();

  bool apply_move(const Move &move);

  std::optional<Move> find_selected_move(Square destination) const;

  std::optional<Square> mouse_to_square(Vector2 mouse_position) const;

  Rectangle square_to_rectangle(Square square) const;

  bool square_contains_human_piece(Square square);

  void load_font();
  void draw_move_highlights() const;
  void draw_status_overlay() const;

  static Side opposite_side(Side side);
  static const char *side_name(Side side);
};