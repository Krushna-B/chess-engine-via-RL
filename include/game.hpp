#pragma once

#include "board.hpp"
#include "move_list.hpp"
#include <ostream>
#include <vector>

enum class GameStatus {
  ONGOING,
  BLACK_WINS,
  WHITE_WINS,
  DRAW_AGREEMENT,

  BLACK_RESIGNS,
  WHITE_RESIGNS,

  WHITE_RAN_OUT_OF_TIME,
  BLACK_RAN_OUT_OF_TIME,

  STALEMATE,
  DRAW_FIFTY_MOVE,
  DRAW_INSUFFICENT_MATERIAL,

  DRAW_THREE_FOLD_REPITITION,
};

// Print enum's as strings
inline std::ostream &operator<<(std::ostream &os, GameStatus s) {
  switch (s) {
  case GameStatus::ONGOING:
    return os << "Game in progress";
  case GameStatus::BLACK_WINS:
    return os << "Black Wins";
  case GameStatus::WHITE_WINS:
    return os << "White Wins";

  case GameStatus::DRAW_AGREEMENT:
    return os << "Draw by Agreement";

  case GameStatus::BLACK_RESIGNS:
    return os << "White Wins (Black Resigned)";
  case GameStatus::WHITE_RESIGNS:
    return os << "Black Wins (White Resigned)";

  case GameStatus::WHITE_RAN_OUT_OF_TIME:
    return os << "Black Wins (White Timeout)";
  case GameStatus::BLACK_RAN_OUT_OF_TIME:
    return os << "White Wins (Black Timeout)";

  case GameStatus::STALEMATE:
    return os << "Draw due to Stalement";

  case GameStatus::DRAW_FIFTY_MOVE:
    return os << "Draw due to 50 Move Rule";
  case GameStatus::DRAW_INSUFFICENT_MATERIAL:
    return os << "Draw due to Insufficient Material";
  case GameStatus::DRAW_THREE_FOLD_REPITITION:
    return os << "Draw due to 3 Fold Repitition";
  };
  return os << "Unknown Status";
}

class Game {
private:
  Position position{};
  std::vector<Move> moves_history{};
  GameStatus status = GameStatus::ONGOING;

  // Update results checking for end of game and 50 mvoe rule
  void update_result();

public:
  Game();

  GameStatus get_status() const;
  Position get_position() const;

  MoveList get_legal_moves();

  bool play_move(const Move &move);

  void reset();

  std::vector<Move> get_move_history() const;
};
