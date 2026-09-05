#include "board.hpp"
#include "move_generator.hpp"
#include "move_list.hpp"
#include "neural_net.hpp"
#include <cmath>
#include <cstddef>
#include <math.h>
#include <memory>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

const float EXPLORATION_COFFICIENT = static_cast<float>(std::sqrt(2));

// For now used in the MCTS, rng used also in select probability from policy
static thread_local std::mt19937_64 rng{
    42}; // statics variables surive function calls are not created every
         // time anad thread_local agives each thread its own copy

struct Node;
float selection(const Node &child, float exploration_coefficient);
float terminal_value(Position &position);
bool is_terminal(Position &position);

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

/***
Returns the V(s) = evaluation of this leaf state
*/
float monte_carlo_tree_sim(Node &node, int depth = 0) {
  std::string indent(static_cast<std::size_t>(depth) * 2, ' ');

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
      float prior = 1.0f / static_cast<float>(moves_array.size());

      std::unique_ptr<Node> child_node =
          std::make_unique<Node>(child, move, &node, prior);

      // TODO: Update this to policy from neural network

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

  return static_cast<float>(q + exploration_coefficient * child.prior *
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
              << " | P = " << child.prior << " | Q = " << q << '\n';
  }

  std::cout << "Total child visits: " << total_child_visits << '\n';

  std::cout << "============================\n";
}

/***
Run search MCTS on some root node
*/
void run_search(Node &root, int simimlations) {
  for (int i{}; i < simimlations; i++) {
    monte_carlo_tree_sim(root);
  }
}

/**
Policy for choosing next move with Temperature
*/
std::vector<float> root_visit_policy(const Node &root, float temperature) {
  std::vector<float> probabilites(
      root.children.size(),
      0.0f); // Create vector 0.0f init size is number of children

  // Exit if no children
  if (root.children.empty()) {
    return probabilites;
  }

  // Temperature below threshold select just the max visited one all the time
  if (temperature <= 0.001f) {
    u64 best_index = 0;
    for (u64 i{}; i < root.children.size(); i++) {
      if (root.children[i]->number_of_visits >
          root.children[best_index]->number_of_visits) {
        best_index = i;
      }
    }
    probabilites[best_index] = 1.0f;
    return probabilites;
  }

  float total_weight{};
  //
  for (u64 i{}; i < root.children.size(); i++) {
    float visits = static_cast<float>(root.children[i]->number_of_visits);

    probabilites[i] = std::pow(visits, 1.0f / temperature);
    total_weight += probabilites[i];
  }

  if (total_weight == 0.0f) {
    return probabilites;
  }

  for (auto &probability : probabilites) {
    probability /= total_weight;
  }

  return probabilites;
}

// Sample move from the probabilites
u64 sample_idx(const std::vector<float> &probabilites, std::mt19937_64 &p_rng) {
  std::discrete_distribution<u64> ditribution(probabilites.begin(),
                                              probabilites.end());
  return ditribution(p_rng);
}

// Reuse the select subtree after a move is played
std::unique_ptr<Node> advance_root(std::unique_ptr<Node> old_root,
                                   u64 selected_idx) {
  std::unique_ptr<Node> new_root = std::move(old_root->children[selected_idx]);
  new_root->parent = nullptr;
  return new_root;
}

/***
Enocde the local_policy into the [4762] output to match neural_net output
*/
PolicyArray encode_policy_target(const Node &root,
                                 const std::vector<float> &local_policy) {

  PolicyArray output{};

  if (local_policy.size() != root.children.size()) {
    throw std::invalid_argument("Policy and child counts do not match up");
  }
  for (u64 child_idx{}; child_idx < root.children.size(); child_idx++) {
    const auto &child = root.children[child_idx];
    u64 action = encode_move(root.state, child->move_from_parent);

    if (action >= POLICY_SIZE) {
      std::out_of_range("Encoded action is exceeds the Policy Array's size");
    }
    output[action] = local_policy[child_idx];
  }
  return output;
}

/**
Validate all probabilites in our stochastic policy
*/
void validate_policy_target(const PolicyArray &policy) {
  float sum = std::accumulate(policy.begin(), policy.end(), 0.0f);

  std::size_t nonzero_actions = 0;

  for (float probability : policy) {
    if (!std::isfinite(probability)) {
      throw std::runtime_error("Policy contains NaN or infinity");
    }

    if (probability < 0.0f) {
      throw std::runtime_error("Policy contains a negative probability");
    }

    if (probability > 0.0f) {
      ++nonzero_actions;
    }
  }

  if (std::abs(sum - 1.0f) > 0.0001f) {
    throw std::runtime_error("Policy probabilities do not sum to one");
  }

  std::cout << "Policy sum: " << sum << '\n';

  std::cout << "Nonzero actions: " << nonzero_actions << '\n';

  std::cout << "Policy target passed\n";
}

// int main() {
//   Position starting_position{};
//   starting_position.set_starting_position();

//   auto root = std::make_unique<Node>(starting_position);

//   constexpr int NUM_SIMULATIONS = 100;
//   constexpr float TEMPERATURE = 0.9f;
//   constexpr int MAX_PLAYS = 512;
//   int plays{};

//   while (!is_terminal(root->state) && plays < MAX_PLAYS) {

//     std::cout << "\n========== REAL MOVE " << plays + 1 << "
//     ==========\n";

//     // Run imaginary MCTS from the current position
//     run_search(*root, NUM_SIMULATIONS);

//     // Convert child nodes into probabilites
//     std::vector<float> policy = root_visit_policy(*root, TEMPERATURE);

//     // Randomly sample from one of these probabilites
//     u64 selected_idx = sample_idx(policy, rng);

//     // Get the move before destroying root
//     Move played_move = root->children[selected_idx]->move_from_parent;
//     std::cout << "Selected child: " << selected_idx << '\n';

//     // New root position
//     root = advance_root(std::move(root), selected_idx);

//     ++plays;
//   }
//   std::cout << "\n========== GAME OVER ==========\n";
//   std::cout << "Total plies: " << plays << '\n';

//   if (root->state.is_checkmate()) {
//     std::cout << "Game ended by checkmate.\n";
//     std::cout << "The player whose turn it is has lost.\n";
//   } else if (root->state.is_stalemate()) {
//     std::cout << "Game ended by stalemate.\n";
//   } else if (root->state.is_draw()) {
//     std::cout << "Game ended in a draw.\n";
//   } else if (plays >= MAX_PLAYS) {
//     std::cout << "Maximum game length reached; treating as draw.\n";
//   }
// };
