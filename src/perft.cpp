#include "board.hpp"
#include "move_generator.hpp"
#include "move_list.hpp"
#include <chrono>
#include <functional>
#include <iostream>

std::string DEPTH;

void perft_driver(int depth, Position &position);

void profile(std::function<void()> func);

// Finding the leaf nodes ( number of positions raeched during the test at
// some depth)
std::uint64_t nodes = 0;

int main() {
  Position position{};
  if (!position.set_from_fen(
          "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1")) {
    std::cout << "Could not parse Kiwipete FEN\n";
    return 1;
  }

  nodes = 0;

  std::cin >> DEPTH;

  profile([&]() { perft_driver(std::stoi(DEPTH), position); });

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
  generate_legal_moves(moves, position);
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