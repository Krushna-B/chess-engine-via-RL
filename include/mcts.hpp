#pragma once

#include "board.hpp"
#include "move_list.hpp"
#include "neural_net.hpp"
#include <memory>
#include <random>
#include <vector>

extern const float EXPLORATION_COFFICIENT;

struct Node {
  Position state{};
  Move move_from_parent{};

  int number_of_visits{}; // N(a,s)
  float prior = 0.0f;     // P(s,a)
  float value_sum{};
  bool expanded{};
  Node *parent{};
  std::vector<std::unique_ptr<Node>> children;

  Node() = default;
  explicit Node(const Position &starting_state) : state{starting_state} {}

  Node(const Position &starting_state, const Move &move, Node *parent_node,
       float prior)
      : state(starting_state), move_from_parent(move), prior(prior),
        parent(parent_node) {}

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
bool is_terminal(Position &position);
float terminal_value(Position &position);
void run_search(Node &root, int simimlations);
std::vector<float> root_visit_policy(const Node &root, float temperature);
u64 sample_idx(const std::vector<float> &probabilites, std::mt19937_64 &p_rng);
std::unique_ptr<Node> advance_root(std::unique_ptr<Node> old_root,
                                   u64 selected_idx);
PolicyArray encode_policy_target(const Node &root,
                                 const std::vector<float> &local_policy);

void validate_policy_target(const PolicyArray &policy);