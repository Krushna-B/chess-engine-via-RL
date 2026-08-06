
#include "board.hpp"
#include "move_generator.hpp"
#include "move_list.hpp"
#include "random_engine.hpp"

#include <iostream>
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

// Match a UCI string (e2e4, e7e8q, e1g1, ...) against the position's legal
// moves
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

int main() {
  std::ios::sync_with_stdio(false);

  Position pos{};
  pos.set_starting_position();
  RandomEngine engine{};

  std::string line;
  while (std::getline(std::cin, line)) {
    std::istringstream ss(line);
    std::string cmd;
    ss >> cmd;

    if (cmd == "uci") {
      std::cout << "id name RandomMoveChessEngine\n";
      std::cout << "id author Krushna Bhanushali\n";
      std::cout << "uciok\n";
    } else if (cmd == "isready") {
      std::cout << "readyok\n";
    } else if (cmd == "ucinewgame") {
      pos.set_starting_position();
    } else if (cmd == "position") {
      handle_position(pos, ss);
    } else if (cmd == "go") {
      // TODO(search): parse ss for wtime/btime/winc/binc/movetime/depth here
      // and budget search time. For a random mover we just reply instantly.
      std::optional<Move> best = engine.choose_move(pos);
      std::cout << "bestmove " << (best ? move_to_uci(*best) : "0000") << "\n";
    } else if (cmd == "quit") {
      break;
    }

    std::cout.flush(); // UCI is line-base flush so cutechess sees replies
  }
  return 0;
}