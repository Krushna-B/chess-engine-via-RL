#include "game_controller.hpp"

#include "gui.hpp"

#include <iostream>
#include <string>

#ifndef ASSETS_DIR
#define ASSETS_DIR "assets"
#endif

namespace {
constexpr int UI_FONT_SIZE = 22; // base glyph size the font is rasterized at
}

GameController::GameController(Side starting_human_side)
    : human_side{starting_human_side},
      engine_side{opposite_side(starting_human_side)} {
  refresh_legal_moves();
}

void GameController::run() {
  InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Chess Engine");

  SetTargetFPS(60);

  gui::load_assets();
  load_font();

  while (!WindowShouldClose()) {
    update();

    BeginDrawing();
    draw();
    EndDrawing();
  }

  UnloadFont(ui_font);
  gui::unload_assets();
  CloseWindow();
}

void GameController::update() {
  /*
   * W: start a new game as White
   * B: start a new game as Black
   * R: restart using the current color
   */
  if (IsKeyPressed(KEY_W)) {
    start_new_game(Side::WHITE);
    return;
  }

  if (IsKeyPressed(KEY_B)) {
    start_new_game(Side::BLACK);
    return;
  }

  if (IsKeyPressed(KEY_R)) {
    start_new_game(human_side);
    return;
  }

  if (controller_error) {
    return;
  }

  if (game.get_status() != GameStatus::ONGOING) {
    return;
  }

  const Side side_to_move = game.get_position().get_side_to_move();

  if (side_to_move == human_side) {
    engine_turn_start = -1.0;
    handle_human_turn();
  } else {
    handle_engine_turn();
  }
}

void GameController::draw() {
  constexpr Color background{28, 28, 28, 255};

  ClearBackground(background);

  /*
  Cream and dark green style board
   */
  gui::draw(game.get_position());

  draw_move_highlights();
  draw_status_overlay();
}

void GameController::start_new_game(Side new_human_side) {
  human_side = new_human_side;
  engine_side = opposite_side(human_side);

  game.reset();

  controller_error = false;
  engine_turn_start = -1.0;

  clear_selection();
  refresh_legal_moves();
}

void GameController::refresh_legal_moves() {
  if (game.get_status() != GameStatus::ONGOING) {
    legal_moves = MoveList{};
    return;
  }

  legal_moves = game.get_legal_moves();
}

void GameController::handle_human_turn() {
  // Right click cancels the current selection.
  if (IsMouseButtonPressed(MOUSE_BUTTON_RIGHT)) {
    clear_selection();
    return;
  }

  if (!IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
    return;
  }

  const std::optional<Square> clicked_square =
      mouse_to_square(GetMousePosition());

  if (!clicked_square.has_value()) {
    clear_selection();
    return;
  }

  const Square square = *clicked_square;

  /*
   * If a piece is already selected, first check whether
   * the clicked square is one of its legal destinations.
   */
  if (selected_square.has_value()) {
    const std::optional<Move> requested_move = find_selected_move(square);

    if (requested_move.has_value()) {
      apply_move(*requested_move);
      return;
    }

    /*
     * Clicking another friendly piece changes the
     * selected piece.
     */
    if (square_contains_human_piece(square)) {
      select_square(square);
      return;
    }

    clear_selection();
    return;
  }

  // Nothing is currently selected.
  if (square_contains_human_piece(square)) {
    select_square(square);
  }
}

void GameController::handle_engine_turn() {
  if (engine_turn_start < 0.0) {
    engine_turn_start = GetTime();
    return;
  }

  if (GetTime() - engine_turn_start < ENGINE_MOVE_DELAY) {
    return;
  }

  engine_turn_start = -1.0;

  const std::optional<Move> move = engine.choose_move(game.get_position());

  if (!move.has_value()) {
    /*
     * If the status is still ongoing but no move exists,
     * Game::update_status() has a bug.
     */
    std::cerr << "Controller error: game is ongoing, but "
              << "the engine has no legal moves.\n";

    controller_error = true;
    return;
  }

  if (!apply_move(*move)) {
    std::cerr << "Controller error: engine-selected move "
              << "was rejected by Game::play_move().\n";

    controller_error = true;
  }
}

void GameController::select_square(Square square) {
  selected_moves.clear();

  for (const Move &move : legal_moves.get_moves()) {
    if (move.from == square) {
      selected_moves.push_back(move);
    }
  }

  if (selected_moves.empty()) {
    clear_selection();
    return;
  }

  selected_square = square;
}

void GameController::clear_selection() {
  selected_square.reset();
  selected_moves.clear();
}

bool GameController::apply_move(const Move &move) {
  /*
   * Game::play_move() should validate the move and then
   * apply the canonical legal Move object.
   */
  if (!game.play_move(move)) {
    std::cerr << "Game::play_move() rejected a controller move.\n";

    return false;
  }

  clear_selection();
  refresh_legal_moves();

  /*
   * If the next turn belongs to the engine, the timer will
   * begin on the next frame.
   */
  engine_turn_start = -1.0;

  return true;
}

std::optional<Move>
GameController::find_selected_move(Square destination) const {
  const Move *fallback_move = nullptr;

  for (const Move &move : selected_moves) {
    if (move.to != destination) {
      continue;
    }

    /*
     * For promotions there are normally four legal moves
     * with the same source and destination:
     *
     * queen, rook, bishop, knight
     *
     * This controller automatically promotes to a queen,
     * avoiding the need for a promotion menu.
     */
    if (move.promotion_piece == QUEEN) {
      return move;
    }

    if (fallback_move == nullptr) {
      fallback_move = &move;
    }
  }

  if (fallback_move != nullptr) {
    return *fallback_move;
  }

  return std::nullopt;
}

std::optional<Square>
GameController::mouse_to_square(Vector2 mouse_position) const {
  if (mouse_position.x < board_bounds.x || mouse_position.y < board_bounds.y ||
      mouse_position.x >= board_bounds.x + board_bounds.width ||
      mouse_position.y >= board_bounds.y + board_bounds.height) {
    return std::nullopt;
  }

  const float square_width = board_bounds.width / 8.0F;

  const float square_height = board_bounds.height / 8.0F;

  const int screen_file =
      static_cast<int>((mouse_position.x - board_bounds.x) / square_width);

  const int screen_rank =
      static_cast<int>((mouse_position.y - board_bounds.y) / square_height);

  /*
   * The GUI is kept in White's orientation:
   *
   * screen top    = rank 8
   * screen bottom = rank 1
   * screen left   = file a
   * screen right  = file h
   */
  const int file = screen_file;
  const int rank = 7 - screen_rank;

  const int square_index = rank * 8 + file;

  return static_cast<Square>(square_index);
}

Rectangle GameController::square_to_rectangle(Square square) const {
  const int square_index = static_cast<int>(square);

  const int file = square_index % 8;
  const int rank = square_index / 8;

  const int screen_file = file;
  const int screen_rank = 7 - rank;

  const float square_width = board_bounds.width / 8.0F;

  const float square_height = board_bounds.height / 8.0F;

  return Rectangle{
      board_bounds.x + static_cast<float>(screen_file) * square_width,

      board_bounds.y + static_cast<float>(screen_rank) * square_height,

      square_width, square_height};
}

bool GameController::square_contains_human_piece(Square square) {
  Position position = game.get_position();

  return get_bit(position.get_occupancy(human_side), square);
}

void GameController::draw_move_highlights() const {
  constexpr Color selected_color{255, 215, 0, 255};

  constexpr Color destination_color{30, 180, 90, 180};

  if (selected_square.has_value()) {
    const Rectangle selected_rectangle = square_to_rectangle(*selected_square);

    DrawRectangleLinesEx(selected_rectangle, 5.0F, selected_color);
  }

  /*
   * Multiple promotion moves may share one destination.
   * Drawing the same circle several times is harmless, but
   * this check prevents duplicates.
   */
  std::vector<Square> drawn_destinations{};

  for (const Move &move : selected_moves) {
    bool already_drawn = false;

    for (Square destination : drawn_destinations) {
      if (destination == move.to) {
        already_drawn = true;
        break;
      }
    }

    if (already_drawn) {
      continue;
    }

    drawn_destinations.push_back(move.to);

    const Rectangle destination_rectangle = square_to_rectangle(move.to);

    const Vector2 center{
        destination_rectangle.x + destination_rectangle.width / 2.0F,

        destination_rectangle.y + destination_rectangle.height / 2.0F};

    DrawCircleV(center, destination_rectangle.width * 0.13F, destination_color);
  }
}

void GameController::load_font() {
  const std::string bundled = std::string(ASSETS_DIR) + "/fonts/Roboto-Regular.ttf";

  // Prefer the bundled font, fall back to a macOS system font, then raylib's
  // built-in default so the app always has something to draw with.
  const char *macos_font = "/System/Library/Fonts/Supplemental/Arial.ttf";

  if (FileExists(bundled.c_str())) {
    ui_font = LoadFontEx(bundled.c_str(), UI_FONT_SIZE, nullptr, 0);
  } else if (FileExists(macos_font)) {
    ui_font = LoadFontEx(macos_font, UI_FONT_SIZE, nullptr, 0);
  } else {
    ui_font = GetFontDefault();
  }

  SetTextureFilter(ui_font.texture, TEXTURE_FILTER_BILINEAR);
}

void GameController::draw_status_overlay() const {
  constexpr Color bar_color{20, 20, 20, 255};
  constexpr Color text_color{235, 235, 235, 255};
  constexpr Color error_color{225, 90, 90, 255};

  // Bars above and below the board (the board occupies the middle strip).
  DrawRectangle(0, 0, gui::WINDOW_WIDTH, gui::TOP_BAR, bar_color);
  DrawRectangle(0, gui::TOP_BAR + gui::BOARD_SIZE, gui::WINDOW_WIDTH,
                gui::BOTTOM_BAR, bar_color);

  const Side current_side = game.get_position().get_side_to_move();

  // Top bar: matchup on the left, state on the right.
  DrawTextEx(ui_font,
             TextFormat("Human: %s   Engine: %s", side_name(human_side),
                        side_name(engine_side)),
             Vector2{16.0F, 17.0F}, UI_FONT_SIZE, 1.0F, text_color);

  const char *state = nullptr;
  Color state_color = text_color;
  if (controller_error) {
    state = "Controller error - check terminal";
    state_color = error_color;
  } else if (game.get_status() == GameStatus::ONGOING) {
    state = TextFormat("%s to move", side_name(current_side));
  } else {
    state = "Game over";
  }

  const Vector2 state_size =
      MeasureTextEx(ui_font, state, UI_FONT_SIZE, 1.0F);
  DrawTextEx(ui_font, state,
             Vector2{gui::WINDOW_WIDTH - state_size.x - 16.0F, 17.0F},
             UI_FONT_SIZE, 1.0F, state_color);

  // Bottom bar: controls hint.
  DrawTextEx(ui_font, "W: play White    B: play Black    R: restart",
             Vector2{16.0F, gui::TOP_BAR + gui::BOARD_SIZE + 11.0F},
             UI_FONT_SIZE, 1.0F, text_color);
}

Side GameController::opposite_side(Side side) {
  return side == Side::WHITE ? Side::BLACK : Side::WHITE;
}

const char *GameController::side_name(Side side) {
  return side == Side::WHITE ? "White" : "Black";
}