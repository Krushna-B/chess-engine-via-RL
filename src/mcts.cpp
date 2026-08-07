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

struct Node;
float selection(const Node &child, float exploration_coefficient);
float terminal_value(Position &position);
bool is_terminal(Position &position);

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
float monte_carlo_tree_sim(Node &node, int depth = 0) {
  std::string indent(depth * 2, ' ');

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
    std::cout << indent << "[EXPAND] depth=" << depth
              << " visits=" << node.number_of_visits << '\n';

    // Run Neural Net Inferende
    // NetworkOutput output = evaluate(node.state);
    // float value = output.value;

    // For now
    static thread_local std::mt19937 rng{
        42}; // statics variables surive function calls are not created every
             // time anad thread_local agives each thread its own copy

    std::uniform_real_distribution<float> real_dist(-1, 1.0);
    auto value = real_dist(rng);
    std::cout << indent << "V(s) = " << value << '\n';

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
    std::cout << indent << "Created " << moves_array.size() << " children\n";
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
  std::cout << indent << "[SELECT] child " << best_idx
            << " score=" << best_score << '\n';

  // Search this subtree and Explore Moves
  float child_value = monte_carlo_tree_sim(*node.children[best_idx], depth + 1);

  // Flip value's due to side change
  float value = -child_value;

  // Backpropagation
  node.number_of_visits++;
  node.value_sum += value;

  std::cout << indent << "[BACKPROP]"
            << " child_value=" << child_value << " -> parent_value=" << value
            << " N=" << node.number_of_visits << " W=" << node.value_sum
            << " Q=" << node.average_reward() << '\n';
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

bool is_terminal(Position &position) {
  return position.is_checkmate() || position.is_stalemate() ||
         position.is_draw();
}

float terminal_value(Position &position) {
  if (position.is_checkmate()) {
    // Player whose turn it is has been checkmated
    return -1.0f;
  }
  if (position.is_stalemate() || position.is_draw()) {
    // Player whose turn it is has been checkmated
    return 0.0f;
  }
  return 0.0f;
}

void print_root_stats(const Node &root) {
  std::cout << "\n============================\n";
  std::cout << "ROOT MCTS RESULTS\n";
  std::cout << "Root visits: " << root.number_of_visits << '\n';

  int total_child_visits = 0;

  for (std::size_t i = 0; i < root.children.size(); ++i) {
    const Node &child = *root.children[i];

    total_child_visits += child.number_of_visits;

    float q =
        child.number_of_visits == 0
            ? 0.0f
            : -child.value_sum / static_cast<float>(child.number_of_visits);

    std::cout << "Child " << i << " | N = " << child.number_of_visits
              << " | P = " << child.policy << " | Q = " << q << '\n';
  }

  std::cout << "Total child visits: " << total_child_visits << '\n';

  std::cout << "============================\n";
}

int main() {
  Position starting_position{};
  starting_position.set_starting_position();
  Node root{starting_position};
  constexpr int NUM_SIMULATIONS = 100;

  for (int i = 0; i < NUM_SIMULATIONS; ++i) {
    std::cout << "\n\n========== SIMULATION " << i + 1 << " ==========\n";
    float result = monte_carlo_tree_sim(root);

    std::cout << "Simulation returned: " << result << '\n';

    print_root_stats(root);
  }
};