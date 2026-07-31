#include "board.hpp"

int main() {
  Board board{};
  Bitboard wK = board.getBitboard(WHITE_KING);
  printBitBoard(wK);

  set_bit(wK, e2);
  clear_bit(wK, e1);
  printBitBoard(wK);

  return 0;
}
