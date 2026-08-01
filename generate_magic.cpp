#include "board.hpp"
#include <fstream>

using namespace std;

int main() {
  const auto bishop_magics = generate_all_bishop_magics();
  //   const auto rook_magics = generate_all_rook_magics();

  std::ofstream bishop_output("magic_bishop_nums.txt");
  std::ofstream rook_output("magic_rook_nums.txt");

  if (!bishop_output.is_open()) {
    std::cerr << "Could not open bishop output file";
    return 1;
  }

  //   if (!rook_output.is_open()) {
  //     std::cerr << "Could not open rook output file";
  //     return 1;
  //   }

  for (auto &magic : bishop_magics) {
    bishop_output << "    " << magic << "ULL,\n";
    // rook_output << "    " << magic << "ULL,\n";
  }
  bishop_output.close();

  return 0;
}
