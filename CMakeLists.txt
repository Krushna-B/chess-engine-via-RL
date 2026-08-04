cmake_minimum_required(VERSION 3.23)

project(chess-engine)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(raylib CONFIG REQUIRED)

add_executable(main
    board.cpp
    move_generator.cpp
    gui.cpp
    main.cpp
    move_list.cpp)
    
add_executable(generate_magic generate_magic.cpp)
add_executable(perft perft.cpp board.cpp
    move_generator.cpp  move_list.cpp)


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
