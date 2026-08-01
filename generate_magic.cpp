#include "board.hpp"
#include <fstream>

using namespace std;

int main() {
  const auto bishop_magics = generate_all_bishop_magics();

  std::ofstream bishop_output("magic_bishop_nums.txt");
  std::ofstream rook_output("magic_rook_nums.txt");

  if (!bishop_output.is_open()) {
    std::cerr << "Could not open file";
    return 1;
  }

  for (auto &magic : bishop_magics) {
    bishop_output << "    " << magic << "ULL,\n";
  }
  bishop_output.close();

  return 0;
}
