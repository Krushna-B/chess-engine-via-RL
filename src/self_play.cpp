#include "board.hpp"
#include "mcts.hpp"
#include "move_list.hpp"
#include "network_inference.hpp"
#include "neural_net.hpp"
#include "position_encoder.hpp"
#include "training_data.hpp"
#include "training_data_v2.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
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

// Emit an inference/throughput snapshot every this many completed games so
// long runs report progress incrementally instead of only at the end.
const int PROGRESS_EVERY = env_int("METRICS_PROGRESS_EVERY", 50);

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
  std::vector<CompactTrainingExample> examples;
  int positions = 0;
  int plies = 0;
  std::string result; // "white_win" | "black_win" | "draw"
  std::string cause;  // checkmate | stalemate | fifty_move | insufficient |
                      // threefold | limit
  long duration_ms = 0;
};

std::uint8_t compact_castling_rights(const Position &position) {
  std::uint8_t rights = 0;
  if (position.has_castling_rights(WHITE_QUEENSIDE)) rights |= 1;
  if (position.has_castling_rights(WHITE_KINGSIDE)) rights |= 2;
  if (position.has_castling_rights(BLACK_QUEENSIDE)) rights |= 4;
  if (position.has_castling_rights(BLACK_KINGSIDE)) rights |= 8;
  return rights;
}

std::array<std::uint64_t, COMPACT_BITPLANES>
compact_bitplanes(const EncodedPositionHistory &encoded) {
  std::array<std::uint64_t, COMPACT_BITPLANES> planes{};
  for (std::size_t plane = 0; plane < COMPACT_BITPLANES; ++plane) {
    for (std::size_t square = 0; square < POSITION_PLANE_SIZE; ++square) {
      if (encoded[plane * POSITION_PLANE_SIZE + square] != 0.0f) {
        planes[plane] |= std::uint64_t{1} << square;
      }
    }
  }
  return planes;
}

std::vector<CompactPolicyEntry>
compact_policy(const PolicyArray &policy) {
  std::vector<CompactPolicyEntry> entries;
  for (std::size_t index = 0; index < policy.size(); ++index) {
    if (policy[index] > 0.0f) {
      entries.push_back(
          {static_cast<std::uint16_t>(index), policy[index]});
    }
  }
  return entries;
}

CompactTrainingExample make_training_example(const PendingExample &pending,
                                             int game_index, int ply,
                                             Side winning_side,
                                             bool checkmate) {
  const Position &position =
      pending.history.get_position(pending.history.size() - 1);
  CompactTrainingExample example{};
  const EncodedPositionHistory encoded =
      encode_position_history(pending.history);
  example.planes = compact_bitplanes(encoded);
  example.castling_rights = compact_castling_rights(position);
  example.side_to_move = position.get_side_to_move() == BLACK ? 1 : 0;
  example.rule50_count = position.get_halfmove_clock();
  example.policy = compact_policy(pending.policy_target);
  if (checkmate) {
    example.wdl = pending.player_to_move == winning_side
                      ? std::array<float, 3>{1.0f, 0.0f, 0.0f}
                      : std::array<float, 3>{0.0f, 0.0f, 1.0f};
  } else {
    example.wdl = {0.0f, 1.0f, 0.0f};
  }
  example.game_id = static_cast<std::uint64_t>(game_index);
  example.ply = static_cast<std::uint32_t>(ply);
  example.source = TrainingSource::SELF_PLAY;
  return example;
}

GameResult play_self_play_game(NeuralNetwork &network, int game_index) {
  const auto game_start = std::chrono::steady_clock::now();

  Position starting_position{};
  starting_position.set_starting_position();

  auto root = std::make_unique<Node>(starting_position);
  int plays{};

  std::vector<PendingExample> history{};

  bool repetition_draw = false;

  while (!is_terminal(root->state) && plays < MAX_PLAYS) {

    // std::cout << "\n========== REAL MOVE " << plays + 1 << " ==========\n";

    // Run imaginary MCTS from the current position
    run_self_play_search(*root, network, SIMULATIONS, rng);

    // Convert child nodes into probabilites
    std::vector<float> training_policy = root_visit_policy(*root, 1.0f);

    PolicyArray fixed_policy = encode_policy_target(*root, training_policy);
    validate_policy_target(fixed_policy);

    Side player = root->state.get_side_to_move();

    // Save the position and MCTS policy before playing selected move
    history.push_back({root->position_history, fixed_policy, player});

    float move_temperature = temperature_for_play(plays);
    std::vector<float> move_policy = root_visit_policy(*root, move_temperature);

    // Randomly sample from one of these probabilites
    u64 selected_idx = sample_idx(move_policy, rng);

    // std::cout << "Selected child: " << selected_idx << '\n';

    // New root position
    root = advance_root(std::move(root), selected_idx);

    ++plays;

    if (root->is_threefold_repetition()) {
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
  for (std::size_t ply = 0; ply < history.size(); ++ply) {
    game.examples.push_back(make_training_example(
        history[ply], game_index, static_cast<int>(ply), winning_side,
        checkmate));
  }

  game.duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::steady_clock::now() - game_start)
                         .count();
  game.positions = static_cast<int>(game.examples.size());

  return game;
}

// JSONL metrics are appended incrementally (per game + periodic snapshots),
// guarded so worker threads can write concurrently.
std::mutex g_metrics_mutex;

void append_metric_line(const std::string &line) {
  std::lock_guard<std::mutex> lock(g_metrics_mutex);
  std::filesystem::create_directories(
      std::filesystem::path(METRICS_PATH).parent_path());
  std::ofstream out(METRICS_PATH, std::ios::app);
  if (out) {
    out << line << "\n";
  }
}

std::string config_record(int games) {
  std::ostringstream o;
  o << "{\"type\":\"config\",\"run_id\":\"" << RUN_ID
    << "\",\"generation\":" << GENERATION << ",\"games\":" << games
    << ",\"simulations\":" << SIMULATIONS << ",\"max_plays\":" << MAX_PLAYS
    << ",\"concurrency\":" << CONCURRENCY << ",\"batch_size\":" << BATCH_SIZE
    << ",\"batch_timeout_us\":" << BATCH_TIMEOUT_US
    << ",\"temp_exploration_plies\":" << TEMP_EXPLORATION_PLIES
    << ",\"temp_high\":1.0,\"temp_low\":0.0}";
  return o.str();
}

std::string game_record(int game_index, const GameResult &game) {
  std::ostringstream o;
  o << "{\"type\":\"game\",\"run_id\":\"" << RUN_ID
    << "\",\"generation\":" << GENERATION << ",\"game_index\":" << game_index
    << ",\"plies\":" << game.plies << ",\"positions\":" << game.examples.size()
    << ",\"result\":\"" << game.result << "\",\"cause\":\"" << game.cause
    << "\",\"duration_ms\":" << game.duration_ms << "}";
  return o.str();
}

// type is "inference" (final) or "inference_progress" (periodic snapshot).
std::string inference_record(const char *type, const InferenceStats &inf,
                             long elapsed_ms) {
  const double throughput =
      elapsed_ms > 0
          ? static_cast<double>(inf.eval_count) / (elapsed_ms / 1000.0)
          : 0.0;
  std::ostringstream o;
  o << "{\"type\":\"" << type << "\",\"run_id\":\"" << RUN_ID
    << "\",\"generation\":" << GENERATION << ",\"device\":\"" << inf.device
    << "\",\"eval_count\":" << inf.eval_count
    << ",\"batch_count\":" << inf.batch_count
    << ",\"mean_batch\":" << inf.mean_batch
    << ",\"max_batch\":" << inf.max_batch
    << ",\"mean_forward_ms\":" << inf.mean_forward_ms
    << ",\"max_forward_ms\":" << inf.max_forward_ms
    << ",\"mean_wait_ms\":" << inf.mean_wait_ms
    << ",\"max_wait_ms\":" << inf.max_wait_ms
    << ",\"throughput_evals_per_sec\":" << throughput << ",\"batch_buckets\":[";
  for (std::size_t i = 0; i < inf.batch_buckets.size(); ++i) {
    o << inf.batch_buckets[i];
    if (i + 1 < inf.batch_buckets.size()) {
      o << ",";
    }
  }
  o << "]}";
  return o.str();
}

std::string selfplay_record(const std::vector<GameResult> &results,
                            long selfplay_ms) {
  long total_positions = 0;
  long sum_plies = 0;
  int white = 0, black = 0, draws = 0;
  int min_plies = std::numeric_limits<int>::max();
  int max_plies = 0;

  for (const GameResult &game : results) {
    total_positions += game.positions;
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
  }

  if (results.empty()) {
    min_plies = 0;
  }

  const double mean_plies = results.empty()
                                ? 0.0
                                : static_cast<double>(sum_plies) /
                                      static_cast<double>(results.size());

  std::ostringstream o;
  o << "{\"type\":\"selfplay\",\"run_id\":\"" << RUN_ID
    << "\",\"generation\":" << GENERATION << ",\"games\":" << results.size()
    << ",\"positions\":" << total_positions << ",\"white_wins\":" << white
    << ",\"black_wins\":" << black << ",\"draws\":" << draws
    << ",\"min_plies\":" << min_plies << ",\"max_plies\":" << max_plies
    << ",\"mean_plies\":" << mean_plies << ",\"duration_ms\":" << selfplay_ms
    << "}";
  return o.str();
}

std::string progress_record(int completed, int total, long elapsed_ms,
                            std::uint64_t positions) {
  const double games_per_sec =
      elapsed_ms > 0 ? completed / (elapsed_ms / 1000.0) : 0.0;
  std::ostringstream o;
  o << "{\"type\":\"selfplay_progress\",\"run_id\":\"" << RUN_ID
    << "\",\"generation\":" << GENERATION
    << ",\"completed_games\":" << completed << ",\"total_games\":"
    << total << ",\"positions\":" << positions
    << ",\"games_per_sec\":" << games_per_sec
    << ",\"elapsed_ms\":" << elapsed_ms << "}";
  return o.str();
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

    // Config record up front so a partial/interrupted run is still described.
    append_metric_line(config_record(GAMES_PER_SHARD));

    // Worker threads pull the next game index until the shard is full; the
    // concurrency is what keeps the inference queue full enough to batch. Each
    // game's record is streamed to disk as it finishes, with a periodic
    // inference/throughput snapshot, so metrics accrue incrementally.
    std::vector<GameResult> results;
    std::atomic<int> next_game{0};
    std::atomic<int> completed_games{0};
    std::atomic<std::uint64_t> completed_positions{0};
    std::mutex results_mutex;
    std::mutex shard_mutex;
    const auto selfplay_start = std::chrono::steady_clock::now();
    std::filesystem::create_directories(
        std::filesystem::path(SHARD_PATH).parent_path());
    CompactTrainingDataWriter shard_writer(SHARD_PATH);

    auto worker = [&]() {
      while (true) {
        int game_index = next_game.fetch_add(1);
        if (game_index >= GAMES_PER_SHARD) {
          break;
        }

        GameResult game = play_self_play_game(network, game_index);
        {
          std::lock_guard<std::mutex> lock(shard_mutex);
          shard_writer.append(game.examples);
        }
        completed_positions.fetch_add(static_cast<std::uint64_t>(game.positions));
        game.examples.clear();
        const std::string line = game_record(game_index, game);

        int completed;
        {
          std::lock_guard<std::mutex> lock(results_mutex);
          results.push_back(std::move(game));
          completed = completed_games.fetch_add(1) + 1;
        }

        append_metric_line(line);

        if (completed % PROGRESS_EVERY == 0) {
          const long elapsed =
              std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::steady_clock::now() - selfplay_start)
                  .count();
          const double games_per_sec =
              elapsed > 0 ? completed / (elapsed / 1000.0) : 0.0;
          std::cout << "[self-play] " << completed << "/" << GAMES_PER_SHARD
                    << " games, " << completed_positions.load()
                    << " positions, " << games_per_sec << " games/s\n"
                    << std::flush;
          append_metric_line(progress_record(
              completed, GAMES_PER_SHARD, elapsed, completed_positions.load()));
          append_metric_line(
              inference_record("inference_progress", network.stats(), elapsed));
        }
      }
    };

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

    // Final aggregates.
    append_metric_line(selfplay_record(results, selfplay_ms));
    append_metric_line(
        inference_record("inference", network.stats(), selfplay_ms));

    shard_writer.close();

    std::cout << "Self-play: " << results.size() << " games, "
              << shard_writer.example_count()
              << " positions in " << selfplay_ms << " ms\n";

    std::vector<CompactTrainingExample> loaded =
        load_compact_training_examples(SHARD_PATH);

    std::cout << "Original shard examples: " << shard_writer.example_count()
              << '\n';

    std::cout << "Loaded shard examples: " << loaded.size() << '\n';

    if (loaded.size() != shard_writer.example_count()) {
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
