#include "game.hpp"
#include "board.hpp"
#include "move_generator.hpp"
#include "move_list.hpp"
#include <vector>

// Update results checking for end of game and 50 mvoe rule
void Game::update_result() {
  MoveList moves = get_legal_moves();

  // Check for checmkate and stalemate
  if (moves.get_moves().size() == 0) {
    Side side = position.get_side_to_move();

    if (position.is_in_check(side)) {
      status = side == WHITE ? GameStatus::BLACK_WINS : GameStatus::WHITE_WINS;
    } else {
      status = GameStatus::STALEMATE;
    }
  }

  // 50 Move Rule
  if (position.get_halfmove_clock() >= 100) {
    status = GameStatus::DRAW_FIFTY_MOVE;
    return;
  }

  // Check for insufficent material
  if (position.has_insufficent_material()) {
    status = GameStatus::DRAW_INSUFFICENT_MATERIAL;
  }

  return;
}

Game::Game() {
  position.set_starting_position();
  update_result();
}

GameStatus Game::get_status() const { return status; }
Position Game::get_position() const { return position; }

MoveList Game::get_legal_moves() {
  MoveList all_moves{};
  generate_legal_moves(all_moves, position);
  return all_moves;
}

bool Game::play_move(Move &move) {
  if (status != GameStatus::ONGOING) {
    return false;
  }
  MoveList moves_list = get_legal_moves();
  auto moves = moves_list.get_moves();
  bool found = false;

  for (int i = 0; i < moves.size(); ++i) {
    if (moves[i] == move) {
      found = true;
      break;
    }
  }

  if (!found) {
    return false;
  }

  position.make_move(move);
  moves_history.push_back(move);
  update_result();
  return true;
}

std::vector<Move> Game::get_move_history() const { return moves_history; }
