#pragma once

#include "board.hpp"
#include "move_list.hpp"
#include <vector>

enum class GameStatus {
  ONGOING,
  BLACK_WINS,
  WHITE_WINS,
  DRAW,

  STALEMATE,
  DRAW_FIFTY_MOVE,
  DRAW_INSUFFICENT_MATERIAL,

  DRAW_THREE_FOLD_REPITITION,
};

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

  bool play_move(Move &move);

  std::vector<Move> get_move_history() const;
};
