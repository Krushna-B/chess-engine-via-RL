#include "board.hpp"
#include "mcts.hpp"
#include "move_generator.hpp"
#include "move_list.hpp"
#include "network_inference.hpp"
#include "neural_net.hpp"

#include <chrono>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>

static std::string square_str(Square s) {
  char f = static_cast<char>('a' + (static_cast<int>(s) % 8));
  char r = static_cast<char>('1' + (static_cast<int>(s) / 8));
  return {f, r};
}

static char promo_char(Piece p) {
  switch (p) {
  case KNIGHT:
    return 'n';
  case BISHOP:
    return 'b';
  case ROOK:
    return 'r';
  case QUEEN:
    return 'q';
  default:
    return '\0';
  }
}

static std::string move_to_uci(const Move &m) {
  std::string s = square_str(m.from) + square_str(m.to);
  if (m.type == MoveType::PROMOTION || m.type == MoveType::PROMOTION_CAPTURE)
    s += promo_char(m.promotion_piece);
  return s;
}

static std::optional<Move> uci_to_move(const std::string &uci, Position &pos) {
  MoveList list{};
  generate_legal_moves(list, pos);
  for (const Move &m : list.get_moves())
    if (move_to_uci(m) == uci)
      return m;
  return std::nullopt;
}

static void handle_position(Position &pos, std::istringstream &args) {
  std::string token;
  args >> token;

  if (token == "startpos") {
    pos.set_starting_position();
    args >> token; // possibly "moves"
  } else if (token == "fen") {
    std::string fen, part;
    for (int i = 0; i < 6 && args >> part; ++i)
      fen += (i ? " " : "") + part;
    pos.set_from_fen(fen);
    args >> token; // possibly "moves"
  }

  if (token == "moves") {
    std::string mv;
    while (args >> mv)
      if (auto m = uci_to_move(mv, pos))
        pos.make_move(*m);
  }
}

// One forward pass, play the legal move the policy head likes best. This is the
// net's "gut" -- no lookahead. Used when Simulations is 0 or the clock is tiny.
static std::optional<Move> pick_by_policy(Position &pos,
                                          NeuralNetwork &network) {
  MoveList list{};
  generate_legal_moves(list, pos);
  auto moves = list.get_moves();
  if (moves.empty())
    return std::nullopt;

  NetworkOutput out = network.evaluate(pos);
  std::optional<Move> best;
  float best_logit = -std::numeric_limits<float>::infinity();
  for (const Move &m : moves) {
    float logit = out.policy_logits[encode_move(pos, m)];
    if (logit > best_logit) {
      best_logit = logit;
      best = m;
    }
  }
  return best;
}

// Run MCTS from pos, stopping at whichever comes first: the simulation cap or
// the wall-clock budget. Play the most-visited child, the AlphaZero choice.
static std::optional<Move> pick_by_search(Position &pos, NeuralNetwork &network,
                                          int max_sims, long budget_ms) {
  if (is_terminal(pos))
    return std::nullopt;

  Node root(pos);
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < max_sims; ++i) {
    monte_carlo_tree_sim(root, network);
    if (budget_ms > 0) {
      long elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::steady_clock::now() - start)
                         .count();
      if (elapsed >= budget_ms)
        break;
    }
  }

  if (root.children.empty())
    return std::nullopt;

  const Node *best = root.children.front().get();
  for (const auto &child : root.children)
    if (child->number_of_visits > best->number_of_visits)
      best = child.get();
  return best->move_from_parent;
}

// Carve a per-move budget out of the clock. movetime wins outright; otherwise
// spend a small slice of the remaining time plus most of the increment.
static long move_budget_ms(long my_time, long my_inc, long movetime) {
  if (movetime > 0)
    return movetime;
  if (my_time <= 0)
    return 0; // no clock info -> run the full simulation cap
  long budget = my_time / 30 + (my_inc * 8) / 10;
  long cap = my_time - 30; // leave a margin so we don't flag
  return budget > cap ? std::max(cap, 1L) : budget;
}

int main(int argc, char **argv) {
  std::ios::sync_with_stdio(false);

  if (argc != 2) {
    std::cerr << "Usage: engine_neural <model-path>\n";
    return 1;
  }

  // Batch size 1 with no wait: a single engine evaluates one leaf at a time, so
  // there's nothing to batch with -- fire each forward pass immediately.
  NeuralNetwork network(argv[1], 1, 0);

  Position pos{};
  pos.set_starting_position();
  int simulations = 200;

  std::string line;
  while (std::getline(std::cin, line)) {
    std::istringstream ss(line);
    std::string cmd;
    ss >> cmd;

    if (cmd == "uci") {
      std::cout << "id name NeuralChessEngine\n";
      std::cout << "id author Krushna Bhanushali\n";
      std::cout << "option name Simulations type spin default 200 min 0 "
                   "max 100000\n";
      std::cout << "uciok\n";
    } else if (cmd == "isready") {
      std::cout << "readyok\n";
    } else if (cmd == "ucinewgame") {
      pos.set_starting_position();
    } else if (cmd == "setoption") {
      std::string token, name;
      long value = 0;
      while (ss >> token) {
        if (token == "name")
          ss >> name;
        else if (token == "value")
          ss >> value;
      }
      if (name == "Simulations")
        simulations = static_cast<int>(value);
    } else if (cmd == "position") {
      handle_position(pos, ss);
    } else if (cmd == "go") {
      long wtime = 0, btime = 0, winc = 0, binc = 0, movetime = 0;
      std::string token;
      while (ss >> token) {
        if (token == "wtime")
          ss >> wtime;
        else if (token == "btime")
          ss >> btime;
        else if (token == "winc")
          ss >> winc;
        else if (token == "binc")
          ss >> binc;
        else if (token == "movetime")
          ss >> movetime;
      }

      bool white = pos.get_side_to_move() == WHITE;
      long budget = move_budget_ms(white ? wtime : btime,
                                   white ? winc : binc, movetime);

      std::optional<Move> best =
          simulations <= 0
              ? pick_by_policy(pos, network)
              : pick_by_search(pos, network, simulations, budget);

      std::cout << "bestmove " << (best ? move_to_uci(*best) : "0000") << "\n";
    } else if (cmd == "quit") {
      break;
    }

    std::cout.flush();
  }
  return 0;
}
