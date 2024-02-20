#include <MappinGen/Grammar.hpp>
#include <MappinGen/MappinGen.hpp>

int main() {
  GrammarParser g_parser("examples/test.grammar");
  std::unique_ptr<grammar::Grammar> g = g_parser.parseGrammar();
  g->print();
  return 0;
};
