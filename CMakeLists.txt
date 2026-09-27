cmake_minimum_required(VERSION 3.23)

project(chess-engine)
enable_testing()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# The raylib GUI target is desktop-only. Disable it for headless builds
option(BUILD_GUI "Build the raylib GUI (main) target" ON)
# LibTorch is not needed for move generation, perft, or the random UCI engine
option(BUILD_ML "Build LibTorch-based self-play and neural-engine targets" ON)

if(BUILD_ML)
    find_package(Torch REQUIRED)
    find_package(Threads REQUIRED)
endif()

#Compiler Flags
set(COMMON_COMPILE_OPTIONS
    -Wall
    -Wextra
    -Wpedantic
    -Wconversion
    -Wshadow
)

# Raise the constexpr evaluation limit. The flag name differs by compiler:
# Clang (local/macOS) uses -fconstexpr-steps, GCC (Linux/Modal) uses
# -fconstexpr-ops-limit
# Pulled out on its own so Torch targets (which don't take the warning flags)
# can still raise the limit without inheriting the rest of COMMON_COMPILE_OPTIONS.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CONSTEXPR_STEPS_OPTION -fconstexpr-steps=100000000)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CONSTEXPR_STEPS_OPTION -fconstexpr-ops-limit=100000000)
endif()
list(APPEND COMMON_COMPILE_OPTIONS ${CONSTEXPR_STEPS_OPTION})

if(BUILD_GUI)
    find_package(raylib CONFIG REQUIRED)

    add_executable(main
        src/board.cpp
        src/move_generator.cpp
        src/gui.cpp
        src/main.cpp
        src/neural_net.cpp
        src/move_list.cpp
        src/game.cpp
        src/game_controller.cpp
        src/random_engine.cpp)

    target_include_directories(main PRIVATE include)
    target_link_libraries(main PRIVATE raylib)

    #Asset path (pieces/fonts) baked in so it resolves regardless of cwd
    target_compile_definitions(main PRIVATE ASSETS_DIR="${CMAKE_SOURCE_DIR}/assets")
    target_compile_options(main PRIVATE ${COMMON_COMPILE_OPTIONS})
endif()

add_executable(generate_magic tools/generate_magic.cpp)
add_executable(perft tools/perft.cpp src/board.cpp
    src/move_generator.cpp  src/move_list.cpp)
add_executable(engine src/uci.cpp src/board.cpp
    src/move_generator.cpp src/move_list.cpp src/random_engine.cpp)
add_executable(position_history_test tests/position_history_test.cpp src/board.cpp
    src/move_generator.cpp src/move_list.cpp)
add_executable(position_encoder_test tests/position_encoder_test.cpp
    src/position_encoder.cpp src/board.cpp src/move_generator.cpp
    src/move_list.cpp)
add_executable(policy_encoder_test tests/policy_encoder_test.cpp
    src/policy_encoder.cpp src/board.cpp src/move_generator.cpp
    src/move_list.cpp)
add_executable(training_data_v2_test tests/training_data_v2_test.cpp
    src/training_data_v2.cpp)
add_executable(game_repetition_test tests/game_repetition_test.cpp src/game.cpp
    src/board.cpp src/move_generator.cpp src/move_list.cpp)


# add_executable(monte_carlo_tree_search
#     src/mcts.cpp
#     src/board.cpp
#     src/move_generator.cpp
#     src/move_list.cpp)

#Header search path
target_include_directories(generate_magic PRIVATE include)
target_include_directories(perft PRIVATE include)
target_include_directories(engine PRIVATE include)
target_include_directories(position_history_test PRIVATE include)
target_include_directories(position_encoder_test PRIVATE include)
target_include_directories(policy_encoder_test PRIVATE include)
target_include_directories(training_data_v2_test PRIVATE include)
target_include_directories(game_repetition_test PRIVATE include)
# target_include_directories(monte_carlo_tree_search  PRIVATE include)
target_compile_options(generate_magic PRIVATE ${COMMON_COMPILE_OPTIONS})
target_compile_options(perft PRIVATE ${COMMON_COMPILE_OPTIONS})
target_compile_options(engine PRIVATE ${COMMON_COMPILE_OPTIONS})
target_compile_options(position_history_test PRIVATE ${COMMON_COMPILE_OPTIONS})
target_compile_options(position_encoder_test PRIVATE ${COMMON_COMPILE_OPTIONS})
target_compile_options(policy_encoder_test PRIVATE ${COMMON_COMPILE_OPTIONS})
target_compile_options(training_data_v2_test PRIVATE ${COMMON_COMPILE_OPTIONS})
target_compile_options(game_repetition_test PRIVATE ${COMMON_COMPILE_OPTIONS})
add_test(NAME position_history COMMAND position_history_test)
add_test(NAME position_encoder COMMAND position_encoder_test)
add_test(NAME policy_encoder COMMAND policy_encoder_test)
add_test(NAME training_data_v2 COMMAND training_data_v2_test)
add_test(NAME game_repetition COMMAND game_repetition_test)
# target_compile_options(monte_carlo_tree_search PRIVATE ${COMMON_COMPILE_OPTIONS})

if(BUILD_ML)
    add_executable(torch_model_test tests/torch_model_test.cpp)
    target_include_directories(torch_model_test SYSTEM PRIVATE ${TORCH_INCLUDE_DIRS})
    target_link_libraries(torch_model_test PRIVATE "${TORCH_LIBRARIES}")
    target_compile_options(torch_model_test PRIVATE ${TORCH_CXX_FLAGS})

    add_executable(mcts_repetition_test
        tests/mcts_repetition_test.cpp
        src/mcts.cpp
        src/neural_net.cpp
        src/board.cpp
        src/network_inference.cpp
        src/move_generator.cpp
        src/move_list.cpp
    )
    target_include_directories(mcts_repetition_test PRIVATE include)
    target_include_directories(mcts_repetition_test SYSTEM PRIVATE ${TORCH_INCLUDE_DIRS})
    target_link_libraries(mcts_repetition_test PRIVATE "${TORCH_LIBRARIES}" Threads::Threads)
    target_compile_options(mcts_repetition_test PRIVATE ${COMMON_COMPILE_OPTIONS} ${TORCH_CXX_FLAGS})
    add_test(NAME mcts_repetition COMMAND mcts_repetition_test)

    add_executable(self_play
        src/self_play.cpp
        src/neural_net.cpp
        src/position_encoder.cpp
        src/mcts.cpp
        src/training_data_v2.cpp
        src/board.cpp
        src/network_inference.cpp
        src/move_generator.cpp
        src/move_list.cpp
    )
    target_include_directories(self_play PRIVATE include)
    target_include_directories(self_play SYSTEM PRIVATE ${TORCH_INCLUDE_DIRS})
    target_link_libraries(self_play PRIVATE "${TORCH_LIBRARIES}" Threads::Threads)
    target_compile_options(self_play PRIVATE ${COMMON_COMPILE_OPTIONS} ${TORCH_CXX_FLAGS})

    # The trained net as a UCI/cutechess opponent. It shares MCTS and inference
    # with self_play but never writes training shards
    add_executable(engine_neural
        src/uci_neural.cpp
        src/neural_net.cpp
        src/mcts.cpp
        src/board.cpp
        src/network_inference.cpp
        src/move_generator.cpp
        src/move_list.cpp
    )
    target_include_directories(engine_neural PRIVATE include)
    target_include_directories(engine_neural SYSTEM PRIVATE ${TORCH_INCLUDE_DIRS})
    target_link_libraries(engine_neural PRIVATE "${TORCH_LIBRARIES}" Threads::Threads)
    target_compile_options(engine_neural PRIVATE ${TORCH_CXX_FLAGS} ${CONSTEXPR_STEPS_OPTION})
endif()
