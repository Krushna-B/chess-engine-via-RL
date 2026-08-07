#include "board.hpp"
#include "move_generator.hpp"
#include "move_list.hpp"
#include <cmath>
#include <math.h>
#include <memory>
#include <random>
#include <utility>
#include <vector>

const float EXPLORATION_COFFICIENT = static_cast<float>(std::sqrt(2));

float selection(float average_reward, float exploration_coefficent,
                float neural_net_policy, int parent_visits, int child_visits);
float terminal_value(const Position &position);
bool is_terminal(const Position &position);

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

/***
Returns the V(s) = evaluation of this leaf state
*/
float monte_carlo_tree_sim(Node &node) {
  // Base Case Reach a new node
  if (is_terminal(node.state)) {
    float value = terminal_value(node.state);

    node.number_of_visits++;
    node.value_sum += value;

    return value;
  }

  // Expansion Step
  //  Make nodes for all possible moves from this state
  if (!node.expanded) {

    // Run Neural Net Inferende
    // NetworkOutput output = evaluate(node.state);
    // float value = output.value;

    // For now
    auto rng = std::mt19937{std::random_device{}()};
    std::uniform_real_distribution<float> real_dist(-1, 1.0);
    auto value = real_dist(rng);

    MoveList moves{};
    generate_legal_moves(moves, node.state);
    auto moves_array = moves.get_moves();
    for (auto &move : moves_array) {
      // Create that state
      Position child = node.state;
      child.make_move(move);
      std::unique_ptr<Node> child_node = std::make_unique<Node>(child);
      child_node->parent = &node;

      // TODO: Update this to policy from neural network
      child_node->policy = 1.0f / static_cast<float>(moves_array.size());

      node.children.push_back(std::move(child_node));
    }
    node.expanded = true;
    node.number_of_visits++;
    node.value_sum += value;
    return value;
  }

  // Already expanded to this node then Selection State
  //  Select a move and explore that
  std::size_t best_idx = 0;
  float best_score = -std::numeric_limits<float>::infinity();
  for (std::size_t i = 0; i < node.children.size(); ++i) {
    float score = selection(*node.children[i], EXPLORATION_COFFICIENT);
    if (score > best_score) {
      best_score = score;
      best_idx = i;
    }
  }

  // Search this subtree and Explore Moves
  float child_value = monte_carlo_tree_sim(*node.children[best_idx]);

  // Flip value's due to side change
  float value = -child_value;

  // Backpropagation
  node.number_of_visits++;
  node.value_sum += value;

  return value;
}

/**
MCTS Selection Algorithm
argmax_a [ Q(a,s) + P(s,a)[ (sqrt(N(s)) / (1 + N(a,s))]]
 */
float selection(const Node &child, float exploration_coefficient) {

  float q =
      (child.number_of_visits == 0
           ? 0.0f
           : -child.value_sum / static_cast<float>(child.number_of_visits));

  return static_cast<float>(q + exploration_coefficient * child.policy *
                                    std::sqrt(child.parent->number_of_visits) /
                                    (1 + child.number_of_visits));
}

bool is_terminal(const Position &position) {
  return position.is_checkmate() || position.is_stalemate() ||
         position.is_draw();
}

float terminal_value(const Position &position) {
  if (position.is_checkmate()) {
    // Player whose turn it is has been checkmated
    return -1.0f;
  }
  if (position.is_stalemate() || positoin.is_draw()) {
    // Player whose turn it is has been checkmated
    return 0.0f;
  }
  return 0.0f;
}