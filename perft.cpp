#include "board.hpp"
#include "move_generator.hpp"
#include "move_list.hpp"
#include <chrono>
#include <functional>

void perft_driver(int depth, Position &position);

void profile(std::function<void()> func);

// Finding the leaf nodes ( number of positions raeched during the test at
// some depth)
std::uint64_t nodes = 0;

int main() {
  Position position{};
  position.set_starting_position();

  const int depth = 6;
  nodes = 0;

  profile([&]() { perft_driver(depth, position); });

  std::cout << "Total Nodes: " << nodes << '\n';

  return 0;
}

// perft driver
void perft_driver(int depth, Position &position) {
  // Escape condition
  if (depth == 0) {
    // increment nodes count
    nodes++;
    return;
  }
  MoveList moves{};
  generate_all_pseudo_moves(moves, position);
  // std::cout << "Depth " << depth << " legal moves: " << moves.get_count()
  //           << '\n';

  auto moves_array = moves.get_moves();
  for (std::size_t i = 0; i < moves.get_count(); ++i) {
    const Move &move = moves_array[i];
    Position child = position;
    if (!child.make_move(move)) {
      // std::cout << "make_move failed: " << move.from << " to " << move.to
      //           << '\n';
      continue;
    }
    perft_driver((depth - 1), child);
  }
}

void profile(std::function<void()> func) {
  auto start = std::chrono::steady_clock::now();
  func();
  auto end = std::chrono::steady_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Time taken: " << duration.count() << " ms\n";
}