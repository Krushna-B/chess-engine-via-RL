#pragma once

#include "neural_net.hpp"

#include <string>
#include <vector>

struct PendingExample {
  EncodedPosition position;
  PolicyArray policy_target;

  /*
    +1 = this positions player eventually won the game
    0 = draw
    -1 = this positions player eventually lost the game
  */
  Side player_to_move;
};

struct TrainingExample {
  EncodedPosition encoded_position;
  PolicyArray policy_target;
  float value_target{};
};

void save_training_examples(const std::string &filename,
                            const std::vector<TrainingExample> &examples);

std::vector<TrainingExample>
load_training_examples(const std::string &filename);