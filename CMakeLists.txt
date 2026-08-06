cmake_minimum_required(VERSION 3.23)

project(chess-engine)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(raylib CONFIG REQUIRED)

add_executable(main
    src/board.cpp
    src/move_generator.cpp
    src/gui.cpp
    src/main.cpp
    src/move_list.cpp
    src/game.cpp
    src/game_controller.cpp
    src/random_engine.cpp)

add_executable(generate_magic src/generate_magic.cpp)
add_executable(perft src/perft.cpp src/board.cpp
    src/move_generator.cpp  src/move_list.cpp)


#Header search path
target_include_directories(main PRIVATE include)
target_include_directories(generate_magic PRIVATE include)
target_include_directories(perft PRIVATE include)

#Linked Libraries
target_link_libraries(main PRIVATE raylib)

#Compiler Flags
set(COMMON_COMPILE_OPTIONS
    -Wall
    -Wextra
    -Wpedantic
    -Wconversion
    -Wshadow
    -fconstexpr-steps=100000000
)

target_compile_options(main PRIVATE ${COMMON_COMPILE_OPTIONS})
target_compile_options(generate_magic PRIVATE ${COMMON_COMPILE_OPTIONS})
target_compile_options(perft PRIVATE ${COMMON_COMPILE_OPTIONS})
