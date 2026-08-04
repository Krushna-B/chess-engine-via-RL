// move_generator.hpp
#pragma once

#include "board.hpp"
#include "move_list.hpp"

/**
Generates all of the legal moves for the current moving side


*/

/**
 * Generates all pseudo-legal moves for the side currently

 * Pseudo-legal moves follow piece movement rules but may leave
 * the moving side's king in check
 */
void generate_all_pseudo_moves(MoveList &moves, Position &position);

/**
 * Generates pawn moves, including:
 * - single pushes
 * - double pushes
 * - captures
 * - promotions
 * - promotion captures
 * - en passant
 */
void generate_pawn_moves(MoveList &moves, const Position &position);

void generate_knight_moves(MoveList &moves, const Position &position);

void generate_bishop_moves(MoveList &moves, const Position &position);

void generate_rook_moves(MoveList &moves, const Position &position);

void generate_queen_moves(MoveList &moves, const Position &position);

/**
 * Generates king moves, including castling
 * These are pseudo-legal king moves
 */
void generate_king_moves(MoveList &moves, const Position &position);

/***
Checks whether the given square is under attack
*/
bool is_square_attacked(Position &position, Square square, Side attacking_side);