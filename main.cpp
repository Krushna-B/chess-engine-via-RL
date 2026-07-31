#include "board.hpp"

int main() {
  Board board{};
  Bitboard wP = board.getBitboard(0);

  board.printBitBoard(wP);
  return 0;
}
