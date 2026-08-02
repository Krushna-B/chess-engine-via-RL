#include <array>
#include <cstddef>
#include <cstdint>
#include <sys/types.h>

struct Move {
  uint8_t from{};
  uint8_t to{};

  Move() = default;

  Move(int from_square, int to_square)
      : from{static_cast<uint8_t>(from_square)}, to {
    static_cast<uint8_t>(to_square)
  }
};

class MoveList {
private:
  static constexpr std::size_t MAX_MOVES = 256;
  std::array<Move, MAX_MOVES> moves{};
  size_t count{};

public:
  void add(Move move) {
    moves[count] = move;
    count++;
  };
  void clear() { moves.fill(Move{}); }
};
