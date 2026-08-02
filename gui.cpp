#include "gui.hpp"

#include <array>
#include <bit>

#include <raylib.h>

// raylib defines WHITE and BLACK as macros
// remove these in this and then redfine colors due to conflict with my enum
#undef WHITE
#undef BLACK

namespace gui {

namespace {

constexpr int SQUARE_SIZE = 80;
constexpr int BOARD_X = 40;
constexpr int BOARD_Y = 40;

constexpr std::array<Piece, 6> PIECES{PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING};

constexpr std::array<Side, 2> SIDES{Side::WHITE, Side::BLACK};

// Custom raylib drawing colors.
constexpr ::Color LIGHT_SQUARE_COLOR{235, 215, 185, 255};

constexpr ::Color DARK_SQUARE_COLOR{165, 115, 85, 255};

constexpr ::Color WHITE_PIECE_COLOR{245, 245, 245, 255};

constexpr ::Color BLACK_PIECE_COLOR{25, 25, 25, 255};

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

const char *piece_text(Side side, Piece piece) {
  if (side == Side::WHITE) {
    switch (piece) {
    case PAWN:
      return "P";

    case KNIGHT:
      return "N";

    case BISHOP:
      return "B";

    case ROOK:
      return "R";

    case QUEEN:
      return "Q";

    case KING:
      return "K";

    default:
      return "";
    }
  }

  switch (piece) {
  case PAWN:
    return "p";

  case KNIGHT:
    return "n";

  case BISHOP:
    return "b";

  case ROOK:
    return "r";

  case QUEEN:
    return "q";

  case KING:
    return "k";

  default:
    return "";
  }
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

void draw_piece(int square, Side side, Piece piece) {
  const char *text = piece_text(side, piece);

  constexpr int FONT_SIZE = 46;

  const int text_width = MeasureText(text, FONT_SIZE);

  const int x = screen_x(square) + (SQUARE_SIZE - text_width) / 2;

  const int y = screen_y(square) + (SQUARE_SIZE - FONT_SIZE) / 2;

  const ::Color text_color =
      side == Side::WHITE ? WHITE_PIECE_COLOR : BLACK_PIECE_COLOR;

  DrawText(text, x, y, FONT_SIZE, text_color);
}

void draw_pieces(const Position &position) {
  for (const Side side : SIDES) {
    for (const Piece piece : PIECES) {
      Bitboard pieces = position.get_piece(side, piece);

      while (pieces != 0ULL) {
        const int square = static_cast<int>(std::countr_zero(pieces));

        pieces &= pieces - 1;

        draw_piece(square, side, piece);
      }
    }
  }
}

} // namespace

void draw(const Position &position) {
  draw_board();
  draw_pieces(position);
}

} // namespace gui