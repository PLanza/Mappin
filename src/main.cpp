#include <MappinGen/MappinGen.hpp>
#include <MappinUtil/MappinUtil.hpp>
#include <iostream>

int main() {
  GrammarParser g_parser("examples/test.grammar");
  try {
    std::unique_ptr<grammar::Grammar> g = g_parser.parseGrammar();
    g->print();
  } catch (MappinException *e) {
    std::cerr << e->what() << std::endl;
  }
  return 0;
};
