#include "gui.hpp"

#include <array>
#include <bit>
#include <string>

#include <raylib.h>

// raylib defines WHITE and BLACK as macros
// remove these in this and then redfine colors due to conflict with my enum
#undef WHITE
#undef BLACK

#ifndef ASSETS_DIR
#define ASSETS_DIR "assets"
#endif

namespace gui {

namespace {

constexpr std::array<Piece, 6> PIECES{PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING};

constexpr std::array<Side, 2> SIDES{Side::WHITE, Side::BLACK};

// Custom raylib drawing colors.
constexpr ::Color LIGHT_SQUARE_COLOR{253, 254, 223, 255};

constexpr ::Color DARK_SQUARE_COLOR{139, 164, 108, 255};

constexpr ::Color WHITE_PIECE_COLOR{245, 245, 245, 255};

constexpr ::Color BLACK_PIECE_COLOR{25, 25, 25, 255};

constexpr ::Color FULL_TINT{255, 255, 255, 255};

// Loaded piece images, indexed [side][piece]. A texture with id 0 means the
// image could not be loaded, and drawing falls back to a letter.
std::array<std::array<Texture2D, 6>, 2> piece_textures{};

int screen_x(int square) {
  const int file = square % 8;
  return BOARD_X + file * SQUARE_SIZE;
}

int screen_y(int square) {
  const int rank = square / 8;

  // Bitboard rank 0 is rank 1, but the top of the
  // screen should display rank 8.
  return BOARD_Y + (7 - rank) * SQUARE_SIZE;
}

char piece_letter(Side side, Piece piece) {
  constexpr char symbols[6] = {'P', 'N', 'B', 'R', 'Q', 'K'};
  const char c = symbols[piece];
  return side == Side::WHITE ? c : static_cast<char>(c - 'A' + 'a');
}

void draw_board() {
  for (int displayed_rank = 0; displayed_rank < 8; ++displayed_rank) {
    for (int file = 0; file < 8; ++file) {
      const ::Color square_color = ((displayed_rank + file) % 2 == 0)
                                       ? LIGHT_SQUARE_COLOR
                                       : DARK_SQUARE_COLOR;

      DrawRectangle(BOARD_X + file * SQUARE_SIZE,
                    BOARD_Y + displayed_rank * SQUARE_SIZE, SQUARE_SIZE,
                    SQUARE_SIZE, square_color);
    }
  }
}

void draw_letter(int square, Side side, Piece piece) {
  constexpr int FONT_SIZE = 46;

  const char text[2] = {piece_letter(side, piece), '\0'};
  const int text_width = MeasureText(text, FONT_SIZE);

  const int x = screen_x(square) + (SQUARE_SIZE - text_width) / 2;
  const int y = screen_y(square) + (SQUARE_SIZE - FONT_SIZE) / 2;

  const ::Color color =
      side == Side::WHITE ? WHITE_PIECE_COLOR : BLACK_PIECE_COLOR;

  DrawText(text, x, y, FONT_SIZE, color);
}

void draw_piece(int square, Side side, Piece piece) {
  const Texture2D &texture = piece_textures[side][piece];

  if (texture.id == 0) {
    draw_letter(square, side, piece);
    return;
  }

  const Rectangle source{0.0F, 0.0F, static_cast<float>(texture.width),
                         static_cast<float>(texture.height)};

  const Rectangle dest{static_cast<float>(screen_x(square)),
                       static_cast<float>(screen_y(square)),
                       static_cast<float>(SQUARE_SIZE),
                       static_cast<float>(SQUARE_SIZE)};

  DrawTexturePro(texture, source, dest, Vector2{0.0F, 0.0F}, 0.0F, FULL_TINT);
}

void draw_pieces(const Position &position) {
  for (const Side side : SIDES) {
    for (const Piece piece : PIECES) {
      Bitboard pieces = position.get_piece(side, piece);

      while (pieces != 0ULL) {
        const int square = static_cast<int>(
            std::countr_zero(pieces)); // Pop LSB to get every piece position

        pieces &= pieces - 1;

        draw_piece(square, side, piece);
      }
    }
  }
}

} // namespace

void load_assets() {
  constexpr char side_prefix[2] = {'w', 'b'};

  for (const Side side : SIDES) {
    for (const Piece piece : PIECES) {
      std::string path = std::string(ASSETS_DIR) + "/pieces/" +
                         side_prefix[side] + piece_letter(WHITE, piece) + ".png";

      if (!FileExists(path.c_str())) {
        continue; // leaves id 0 -> letter fallback
      }

      Texture2D texture = LoadTexture(path.c_str());
      SetTextureFilter(texture, TEXTURE_FILTER_BILINEAR); // 60px -> 90px
      piece_textures[side][piece] = texture;
    }
  }
}

void unload_assets() {
  for (const Side side : SIDES) {
    for (const Piece piece : PIECES) {
      if (piece_textures[side][piece].id != 0) {
        UnloadTexture(piece_textures[side][piece]);
        piece_textures[side][piece] = Texture2D{};
      }
    }
  }
}

void draw(const Position &position) {
  draw_board();
  draw_pieces(position);
}

} // namespace gui
