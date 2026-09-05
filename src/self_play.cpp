#include "board.hpp"
#include "mcts.hpp"
#include "move_list.hpp"
#include "neural_net.hpp"
#include <random>
#include <vector>

constexpr int SIMULATIONS = 800;
constexpr float TEMPERATURE = 0.9f;
constexpr int MAX_PLAYS = 512;

static thread_local std::mt19937_64 rng{42};

struct PendingExmple {
  Position position;
  std::vector<float> policy_target;
  Side player_to_move;
};

struct TrainingExmaple {
  Position position;
  std::vector<float> policy_target;
  float value_target;
};

int main() {
  Position starting_position{};
  starting_position.set_starting_position();

  auto root = std::make_unique<Node>(starting_position);
  int plays{};

  while (!is_terminal(root->state) && plays < MAX_PLAYS) {

    std::cout << "\n========== REAL MOVE " << plays + 1 << " ==========\n";

    // Run imaginary MCTS from the current position
    run_search(*root, SIMULATIONS);

    // Convert child nodes into probabilites
    std::vector<float> local_policy = root_visit_policy(*root, TEMPERATURE);

    PolicyArray fixed_policy = encode_policy_target(*root, local_policy);
    validate_policy_target(fixed_policy);

    // Randomly sample from one of these probabilites
    u64 selected_idx = sample_idx(local_policy, rng);

    // Get the move before destroying root
    Move played_move = root->children[selected_idx]->move_from_parent;
    std::cout << "Selected child: " << selected_idx << '\n';

    // New root position
    root = advance_root(std::move(root), selected_idx);

    ++plays;
  }
  std::cout << "\n========== GAME OVER ==========\n";
  std::cout << "Total plies: " << plays << '\n';

  if (root->state.is_checkmate()) {
    std::cout << "Game ended by checkmate.\n";
    std::cout << "The player whose turn it is has lost.\n";
  } else if (root->state.is_stalemate()) {
    std::cout << "Game ended by stalemate.\n";
  } else if (root->state.is_draw()) {
    std::cout << "Game ended in a draw.\n";
  } else if (plays >= MAX_PLAYS) {
    std::cout << "Maximum game length reached; treating as draw.\n";
  }
  return 0;
}

// int main() {
//   Position position{};
//   position.set_starting_position();
//   position.set_from_fen("r3k2r/p1ppqpb1/bn2pnp1/2pP4/"
//                         "1p2P3/2N2N2/PPQBBPPP/R3K2R "
//                         "b KQkq - 0 1");
//   validate_move_encoding(position);

//   return 0;
// }
