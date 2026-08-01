#include "attacks.hpp"

int main() {

  for (auto square : BISHOP_TABLES.BISHOP_LUT) {
    for (auto &occupancy : square) {
      printBitBoard(occupancy);
    }
  }

  return 0;
}
