#include "board.hpp"
#include "mcts.hpp"
#include "move_list.hpp"
#include "network_inference.hpp"
#include "neural_net.hpp"
#include "training_data.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

static int env_int(const char *name, int fallback) {
  const char *value = std::getenv(name);
  return value != nullptr ? std::atoi(value) : fallback;
}

static std::string env_str(const char *name, const char *fallback) {
  const char *value = std::getenv(name);
  return value != nullptr ? std::string(value) : std::string(fallback);
}

const std::string SHARD_PATH = env_str(
    "SELFPLAY_SHARD_PATH", "artifacts/selfplay/neural_selfplay_shard_0001.bin");

const int SIMULATIONS = env_int("SELFPLAY_SIMULATIONS", 300);
const int MAX_PLAYS = env_int("SELFPLAY_MAX_PLAYS", 512);

// How many games run at once, This is what fills the inference batch with N
// games
const int CONCURRENCY = env_int("SELFPLAY_CONCURRENCY", 512);
const int BATCH_SIZE = env_int("SELFPLAY_BATCH", 256);
const int BATCH_TIMEOUT_US = env_int("SELFPLAY_BATCH_TIMEOUT_US", 1000);

const int GENERATION = env_int("SELFPLAY_GENERATION", 0);
const std::string RUN_ID = env_str("RUN_ID", "adhoc");
const std::string METRICS_PATH =
    env_str("METRICS_PATH", "artifacts/metrics/metrics.jsonl");

// Temperature schedule (mirrors temperature_for_play), logged in config.
constexpr int TEMP_EXPLORATION_PLIES = 30;

static thread_local std::mt19937_64 rng{std::random_device{}()};

Side opposite_side(Side side) {
  return side == Side::WHITE ? Side::BLACK : Side::WHITE;
}
float temperature_for_play(int play) {
  if (play < TEMP_EXPLORATION_PLIES) {
    return 1.0f;
  }

  return 0.0f;
}

struct GameResult {
  std::vector<TrainingExample> examples;
  int plies = 0;
  std::string result; // "white_win" | "black_win" | "draw"
  std::string cause;  // checkmate | stalemate | fifty_move | insufficient |
                      // threefold | limit
  long duration_ms = 0;
};

GameResult play_self_play_game(NeuralNetwork &network) {
  const auto game_start = std::chrono::steady_clock::now();

  Position starting_position{};
  starting_position.set_starting_position();

  auto root = std::make_unique<Node>(starting_position);
  int plays{};

  std::vector<PendingExample> history{};

  // Real-game position hashes for threefold-repetition detection.
  std::unordered_map<u64, int> position_counts;
  position_counts[root->state.hash()] = 1;
  bool repetition_draw = false;

  while (!is_terminal(root->state) && plays < MAX_PLAYS) {

    // std::cout << "\n========== REAL MOVE " << plays + 1 << " ==========\n";

    // Run imaginary MCTS from the current position
    run_self_play_search(*root, network, SIMULATIONS, rng);

    // Convert child nodes into probabilites
    std::vector<float> training_policy = root_visit_policy(*root, 1.0f);

    PolicyArray fixed_policy = encode_policy_target(*root, training_policy);
    validate_policy_target(fixed_policy);

    EncodedPosition enocded_position = encode_position(root->state);
    Side player = root->state.get_side_to_move();

    // Save the position and MCTS policy before playing selected move
    history.push_back({enocded_position, fixed_policy, player});

    float move_temperature = temperature_for_play(plays);
    std::vector<float> move_policy = root_visit_policy(*root, move_temperature);

    // Randomly sample from one of these probabilites
    u64 selected_idx = sample_idx(move_policy, rng);

    // std::cout << "Selected child: " << selected_idx << '\n';

    // New root position
    root = advance_root(std::move(root), selected_idx);

    ++plays;

    // Threefold repetition ends the game as a draw.
    if (++position_counts[root->state.hash()] >= 3) {
      repetition_draw = true;
      break;
    }
  }
  bool checkmate = root->state.is_checkmate();
  bool stalemate = root->state.is_stalemate();
  bool fifty_move = root->state.get_halfmove_clock() >= 100;
  bool insufficient = root->state.has_insufficent_material();
  bool reached_limit = plays >= MAX_PLAYS && !checkmate && !stalemate &&
                       !fifty_move && !insufficient && !repetition_draw;

  Side losing_side = root->state.get_side_to_move();
  Side winning_side = opposite_side(losing_side);

  GameResult game{};
  game.plies = plays;
  if (checkmate) {
    game.result = winning_side == Side::WHITE ? "white_win" : "black_win";
    game.cause = "checkmate";
  } else {
    game.result = "draw";
    game.cause = repetition_draw ? "threefold"
                 : stalemate     ? "stalemate"
                 : fifty_move    ? "fifty_move"
                 : insufficient  ? "insufficient"
                                 : "limit";
  }

  game.examples.reserve(history.size());
  for (const PendingExample &pending : history) {
    float value_target = 0.0f;
    if (checkmate) {
      value_target = pending.player_to_move == winning_side ? 1.0f : -1.0f;
    }
    game.examples.push_back(
        {pending.position, pending.policy_target, value_target});
  }

  game.duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::steady_clock::now() - game_start)
                         .count();

  return game;
}

// Append the full telemetry for this generation as JSONL records.
void write_metrics(const std::vector<GameResult> &results,
                   const InferenceStats &inf, long selfplay_ms) {
  std::filesystem::create_directories(
      std::filesystem::path(METRICS_PATH).parent_path());

  std::ofstream out(METRICS_PATH, std::ios::app);
  if (!out) {
    return;
  }

  out << "{\"type\":\"config\",\"run_id\":\"" << RUN_ID
      << "\",\"generation\":" << GENERATION
      << ",\"games\":" << results.size() << ",\"simulations\":" << SIMULATIONS
      << ",\"max_plays\":" << MAX_PLAYS << ",\"concurrency\":" << CONCURRENCY
      << ",\"batch_size\":" << BATCH_SIZE
      << ",\"batch_timeout_us\":" << BATCH_TIMEOUT_US
      << ",\"temp_exploration_plies\":" << TEMP_EXPLORATION_PLIES
      << ",\"temp_high\":1.0,\"temp_low\":0.0}\n";

  long total_positions = 0;
  long sum_plies = 0;
  int white = 0, black = 0, draws = 0;
  int min_plies = std::numeric_limits<int>::max();
  int max_plies = 0;

  for (std::size_t i = 0; i < results.size(); ++i) {
    const GameResult &game = results[i];
    total_positions += static_cast<long>(game.examples.size());
    sum_plies += game.plies;
    min_plies = std::min(min_plies, game.plies);
    max_plies = std::max(max_plies, game.plies);
    if (game.result == "white_win") {
      ++white;
    } else if (game.result == "black_win") {
      ++black;
    } else {
      ++draws;
    }

    out << "{\"type\":\"game\",\"run_id\":\"" << RUN_ID
        << "\",\"generation\":" << GENERATION
        << ",\"game_index\":" << i << ",\"plies\":" << game.plies
        << ",\"positions\":" << game.examples.size() << ",\"result\":\""
        << game.result << "\",\"cause\":\"" << game.cause
        << "\",\"duration_ms\":" << game.duration_ms << "}\n";
  }

  if (results.empty()) {
    min_plies = 0;
  }

  const double mean_plies = results.empty()
                                ? 0.0
                                : static_cast<double>(sum_plies) /
                                      static_cast<double>(results.size());

  out << "{\"type\":\"selfplay\",\"run_id\":\"" << RUN_ID
      << "\",\"generation\":" << GENERATION
      << ",\"games\":" << results.size() << ",\"positions\":" << total_positions
      << ",\"white_wins\":" << white << ",\"black_wins\":" << black
      << ",\"draws\":" << draws << ",\"min_plies\":" << min_plies
      << ",\"max_plies\":" << max_plies << ",\"mean_plies\":" << mean_plies
      << ",\"duration_ms\":" << selfplay_ms << "}\n";

  const double throughput =
      selfplay_ms > 0
          ? static_cast<double>(inf.eval_count) / (selfplay_ms / 1000.0)
          : 0.0;

  out << "{\"type\":\"inference\",\"run_id\":\"" << RUN_ID
      << "\",\"generation\":" << GENERATION
      << ",\"device\":\"" << inf.device
      << "\",\"eval_count\":" << inf.eval_count
      << ",\"batch_count\":" << inf.batch_count
      << ",\"mean_batch\":" << inf.mean_batch
      << ",\"max_batch\":" << inf.max_batch
      << ",\"mean_forward_ms\":" << inf.mean_forward_ms
      << ",\"max_forward_ms\":" << inf.max_forward_ms
      << ",\"mean_wait_ms\":" << inf.mean_wait_ms
      << ",\"max_wait_ms\":" << inf.max_wait_ms
      << ",\"throughput_evals_per_sec\":" << throughput
      << ",\"batch_buckets\":[";
  for (std::size_t i = 0; i < inf.batch_buckets.size(); ++i) {
    out << inf.batch_buckets[i];
    if (i + 1 < inf.batch_buckets.size()) {
      out << ",";
    }
  }
  out << "]}\n";
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

    // Load the model exactly once. The batching server owns it
    NeuralNetwork network(model_path, BATCH_SIZE, BATCH_TIMEOUT_US);

    std::cout << "Model loaded successfully\n";

    const int GAMES_PER_SHARD = env_int("SELFPLAY_GAMES", 2000);

    // Worker threads pull the next game index until the shard is full; the
    // concurrency is what keeps the inference queue full enough to batch.
    std::vector<GameResult> results;
    std::atomic<int> next_game{0};
    std::mutex results_mutex;

    auto worker = [&]() {
      while (true) {
        int game_index = next_game.fetch_add(1);
        if (game_index >= GAMES_PER_SHARD) {
          break;
        }

        GameResult game = play_self_play_game(network);

        std::lock_guard<std::mutex> lock(results_mutex);
        results.push_back(std::move(game));
      }
    };

    const auto selfplay_start = std::chrono::steady_clock::now();
    {
      const int num_threads = std::min(CONCURRENCY, GAMES_PER_SHARD);

      std::vector<std::thread> threads;
      threads.reserve(static_cast<std::size_t>(num_threads));
      for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker);
      }
      for (std::thread &thread : threads) {
        thread.join();
      }
    }
    const long selfplay_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - selfplay_start)
            .count();

    // Capture all telemetry first (reads per-game example counts), then merge.
    write_metrics(results, network.stats(), selfplay_ms);

    std::vector<TrainingExample> shard;
    shard.reserve(static_cast<std::size_t>(GAMES_PER_SHARD) * 200);
    for (GameResult &game : results) {
      shard.insert(shard.end(), std::make_move_iterator(game.examples.begin()),
                   std::make_move_iterator(game.examples.end()));
    }

    std::filesystem::create_directories(
        std::filesystem::path(SHARD_PATH).parent_path());
    save_training_examples(SHARD_PATH, shard);

    std::cout << "Self-play: " << results.size() << " games, " << shard.size()
              << " positions in " << selfplay_ms << " ms\n";

    std::vector<TrainingExample> loaded = load_training_examples(SHARD_PATH);

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
