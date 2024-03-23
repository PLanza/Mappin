#include <generate/grammar/llgrammar.hpp>
#include <generate/grammar_parser.hpp>
#include <iostream>
#include <parse/inet.hpp>
#include <parse/nodes.hpp>
#include <parse/parser.hpp>
#include <parse/draw.hpp>
#include <util/util.hpp>

int main() {
  std::unique_ptr<grammar::Grammar> g;
  try {
    GrammarParser<grammar::LLGrammar> g_parser("examples/test.grammar");
    g = g_parser.parseGrammar();
    g->printGrammar();
    g->makeParseTable();
    g->printParseTable();
    g->generateStackActions();
    g->printStackActions();
  } catch (MappinException *e) {
    std::cerr << e->what() << std::endl;
    return -1;
  }

  inet::init();
  std::vector<grammar::Token> tokens = g->stringToTokens("B B X Z Z C C Y");
  inet::create_parser_network(g->getStackActions(), tokens);
  
  inet::drawNetwork();

  while (!inet::interactions.empty()) {
    inet::interact();
  }

  return 0;
};
