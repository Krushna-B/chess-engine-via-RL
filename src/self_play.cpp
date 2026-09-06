#include "board.hpp"
#include "mcts.hpp"
#include "move_list.hpp"
#include "neural_net.hpp"
#include "training_data.hpp"
#include <chrono>
#include <random>
#include <stdexcept>
#include <vector>

constexpr int SIMULATIONS = 300;
constexpr float TEMPERATURE = 0.9f;
constexpr int MAX_PLAYS = 512;

static thread_local std::mt19937_64 rng{std::random_device{}()};

Side opposite_side(Side side) {
  return side == Side::WHITE ? Side::BLACK : Side::WHITE;
}

std::vector<TrainingExample> play_self_play_game(NeuralNetwork &network) {
  Position starting_position{};
  starting_position.set_starting_position();

  auto root = std::make_unique<Node>(starting_position);
  int plays{};

  std::vector<PendingExample> history{};

  while (!is_terminal(root->state) && plays < MAX_PLAYS) {

    // std::cout << "\n========== REAL MOVE " << plays + 1 << " ==========\n";

    // Run imaginary MCTS from the current position
    run_search(*root, network, SIMULATIONS);

    // Convert child nodes into probabilites
    std::vector<float> local_policy = root_visit_policy(*root, TEMPERATURE);

    PolicyArray fixed_policy = encode_policy_target(*root, local_policy);
    validate_policy_target(fixed_policy);

    EncodedPosition enocded_position = encode_position(root->state);
    Side player = root->state.get_side_to_move();

    // Save the position and MCTS policy before playing selected move
    history.push_back({enocded_position, fixed_policy, player});

    // Randomly sample from one of these probabilites
    u64 selected_idx = sample_idx(local_policy, rng);

    // Get the move before destroying root
    Move played_move = root->children[selected_idx]->move_from_parent;
    // std::cout << "Selected child: " << selected_idx << '\n';

    // New root position
    root = advance_root(std::move(root), selected_idx);

    ++plays;
  }
  bool checkmate = root->state.is_checkmate();

  bool stalemate = root->state.is_stalemate();
  bool rule_draw = root->state.is_draw();
  bool reached_limit =
      plays >= MAX_PLAYS && !checkmate && !stalemate && !rule_draw;

  bool draw = stalemate || rule_draw || reached_limit;
  Side losing_side = root->state.get_side_to_move();

  Side winning_side = opposite_side(losing_side);

  std::vector<TrainingExample> examples{};
  examples.reserve(history.size());

  for (const PendingExample &pending : history) {
    float value_target = 0.0f;
    if (checkmate) {
      value_target = pending.player_to_move == winning_side ? 1.0f : -1.0f;
    } else if (draw) {
      value_target = 0.0f;
    } else {
      throw std::runtime_error("Game ended without a recognized result");
    }
    examples.push_back({pending.position, pending.policy_target, value_target});
  }

  std::cout << "\nGame generated " << examples.size() << " training examples\n";

  return examples;
}

void profile(std::function<void()> func) {
  auto start = std::chrono::steady_clock::now();
  func();
  auto end = std::chrono::steady_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Time taken: " << duration.count() << " ms\n";
}

int main(int argc, char **argv) {

  if (argc != 2) {
    std::cerr << "Usage: self_play <model-path>\n";

    return 1;
  }

  try {
    const std::string model_path = argv[1];

    // Load the model exactly once.
    NeuralNetwork network(model_path);

    std::cout << "Model loaded successfully\n";

    constexpr int GAMES_PER_SHARD = 1;

    std::vector<TrainingExample> shard;
    shard.reserve(GAMES_PER_SHARD * 200);

    profile([&]() {
      for (int game_num = 0; game_num < GAMES_PER_SHARD; ++game_num) {
        // Every game uses the same frozen model.
        std::vector<TrainingExample> game = play_self_play_game(network);

        std::cout << "Game " << game_num + 1 << " generated " << game.size()
                  << " examples\n";

        shard.insert(shard.end(), std::make_move_iterator(game.begin()),
                     std::make_move_iterator(game.end()));
      }

      save_training_examples("neural_selfplay_shard_0001.bin", shard);
    });

    std::vector<TrainingExample> loaded =
        load_training_examples("neural_selfplay_shard_0001.bin");

    std::cout << "Original shard examples: " << shard.size() << '\n';

    std::cout << "Loaded shard examples: " << loaded.size() << '\n';

    if (loaded.size() != shard.size()) {
      throw std::runtime_error("Shard example count mismatch");
    }

    std::cout << "Neural shard save/load test passed\n";

  } catch (const std::exception &error) {
    std::cerr << "Error: " << error.what() << '\n';

    return 1;
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
