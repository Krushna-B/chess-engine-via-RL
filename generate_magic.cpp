#include "board.hpp"
#include <fstream>

using namespace std;

void generate_all_bishop_magics();

int main() {
  generate_all_bishop_magics();
  return 0;
}

// Magic number's for every single square a1,......h8, and write to file
void generate_all_bishop_magics() {
  std::ofstream bishop_output("magic_bishop_nums.txt");
  for (int square{}; square < 64; square++) {
    auto magic = find_bishop_magic(square);
    std::cout << "Magic number for " << square << "is " << magic << std::endl;
    if (!bishop_output.is_open()) {
      std::cerr << "File could not be opened";
      return;
    }
    bishop_output << magic << std::endl;
    bishop_output.close();
  }
}