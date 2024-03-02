#include <generate/grammar_parser.hpp>
#include <iostream>
#include <util/util.hpp>

int main() {
  try {
    GrammarParser g_parser("examples/test.grammar");
    std::unique_ptr<grammar::Grammar> g = g_parser.parseGrammar();
    g->print();
    g->makeParseTable();
  } catch (MappinException *e) {
    std::cerr << e->what() << std::endl;
  }
  return 0;
};
