#include <MappinGen/Grammar.hpp>
#include <MappinGen/MappinGen.hpp>

int main() {
  GrammarParser g_parser("examples/test.grammar");
  grammar::TermOrNonTerm t = grammar::newTerminal("A");
  return 0;
};
