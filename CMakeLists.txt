cmake_minimum_required(VERSION 3.23)

project(chess-engine)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# The raylib GUI (main) target is desktop-only. Disable it for headless
option(BUILD_GUI "Build the raylib GUI (main) target" ON)

find_package(Torch REQUIRED)
find_package(Threads REQUIRED)

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
# -fconstexpr-ops-limit.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    list(APPEND COMMON_COMPILE_OPTIONS -fconstexpr-steps=100000000)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    list(APPEND COMMON_COMPILE_OPTIONS -fconstexpr-ops-limit=100000000)
endif()

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

add_executable(generate_magic src/generate_magic.cpp)
add_executable(perft src/perft.cpp src/board.cpp
    src/move_generator.cpp  src/move_list.cpp)
add_executable(engine src/uci.cpp src/board.cpp
    src/move_generator.cpp src/move_list.cpp src/random_engine.cpp)


add_executable(
    torch_model_test
    src/torch_model_test.cpp
)


# add_executable(monte_carlo_tree_search
#     src/mcts.cpp
#     src/board.cpp
#     src/move_generator.cpp
#     src/move_list.cpp)

add_executable(self_play
    src/self_play.cpp
    src/neural_net.cpp
    src/mcts.cpp
    src/training_data.cpp
    src/board.cpp
    src/network_inference.cpp
    src/move_generator.cpp
    src/move_list.cpp
)

#Header search path
target_include_directories(generate_magic PRIVATE include)
target_include_directories(perft PRIVATE include)
target_include_directories(engine PRIVATE include)
# target_include_directories(monte_carlo_tree_search  PRIVATE include)
target_include_directories(
    torch_model_test
    SYSTEM PRIVATE
    ${TORCH_INCLUDE_DIRS}
)


#Linked Libraries
target_link_libraries(
    torch_model_test
    PRIVATE
    "${TORCH_LIBRARIES}"
)


target_compile_options(generate_magic PRIVATE ${COMMON_COMPILE_OPTIONS})
target_compile_options(perft PRIVATE ${COMMON_COMPILE_OPTIONS})
target_compile_options(engine PRIVATE ${COMMON_COMPILE_OPTIONS})
# target_compile_options(monte_carlo_tree_search PRIVATE ${COMMON_COMPILE_OPTIONS})
target_compile_options(self_play PRIVATE ${COMMON_COMPILE_OPTIONS})
target_compile_options(
    torch_model_test
    PRIVATE
    ${TORCH_CXX_FLAGS}
)

target_include_directories(
    self_play
    PRIVATE
    include
)

target_include_directories(
    self_play
    SYSTEM PRIVATE
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(
    self_play
    PRIVATE
    "${TORCH_LIBRARIES}"
    Threads::Threads
)

target_compile_options(
    self_play
    PRIVATE
    ${TORCH_CXX_FLAGS}
)
