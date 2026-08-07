#pragma once

#include "board.hpp"
#include <memory>
#include <vector>

extern const float EXPLORATION_COFFICIENT;

struct Node {
  Position state{};
  int number_of_visits{}; // N(a,s)
  float policy = 0.0f;    // P(s,a)
  float value_sum{};
  bool expanded{};
  Node *parent{};
  std::vector<std::unique_ptr<Node>> children;

  Node() = default;
  Node(Position &starting_state) : state{starting_state} {};

  /**
  Q(s,a)
   */
  float average_reward() const {
    if (number_of_visits == 0)
      return 0.0f;

    return value_sum / static_cast<float>(number_of_visits);
  }
};

float monte_carlo_tree_sim(Node &node, int depth = 0);
float selection(const Node &child, float exploration_coefficient);
bool is_terminal(const Position &position);
float terminal_value(const Position &position);
