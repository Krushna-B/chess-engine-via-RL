#include "Bitboard.hpp"
#include "attacks.hpp"
#include <bit>

// Helper function to pop the position of first 1 bit (Whihc is position of
// piece)
int pop_lsb(Bitboard &bitboard) {
  const int square = static_cast<int>(std::countr_zero(bitboard));

  bitboard &= bitboard - 1;
  return square;
}
