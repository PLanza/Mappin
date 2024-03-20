#include <generate/grammar/llgrammar.hpp>
#include <generate/grammar_parser.hpp>
#include <iostream>
#include <parse/parse_net.hpp>
#include <parse/inet.hpp>
#include <util/util.hpp>

int main() {
  try {
    GrammarParser<grammar::LLGrammar> g_parser("examples/test.grammar");
    std::unique_ptr<grammar::Grammar> g = g_parser.parseGrammar();
    g->printGrammar();
    g->makeParseTable();
    g->printParseTable();
    g->generateStackActions();
    g->printStackActions();

  } catch (MappinException *e) {
    std::cerr << e->what() << std::endl;
  }

  inet::init();
  while (!inet::interactions.empty()){
    inet::interact();
  }
  
  return 0;
};
