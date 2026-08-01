cmake_minimum_required(VERSION 3.23)

project(chess-engine)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(main main.cpp)
add_executable(generate_magic generate_magic.cpp)



#Compiler Flags
target_compile_options(main PRIVATE
    -Wall -Wextra -Wpedantic -Wconversion -Wshadow)

