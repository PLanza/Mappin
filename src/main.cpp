#include "../include/generate/grammar/llgrammar.hpp"
#include "../include/generate/grammar/lrgrammar.hpp"
#include "../include/generate/grammar_parser.hpp"
#include "../include/generate/util.hpp"
#include "../include/gpu/run.hpp"
#include <iostream>
#include <string>

int main() {
  std::unique_ptr<grammar::Grammar> g;
  try {
    GrammarParser<grammar::LRGrammar> g_parser("examples/test.grammar");
    g = g_parser.parseGrammar();
    g->printGrammar();
    g->makeParseTable();
    g->printParseTable();
    g->generateStackActions();
    g->printStackActions();
  } catch (MappinException *e) {
    const char *e_str = e->what();
    std::cerr << e_str << std::endl;
    delete[] e_str;
    delete e;
    return -1;
  }

  // std::string parseString;
  // std::getline(std::cin, parseString);
  //
  // parse(std::move(g), parseString);
  // inet::init();
  // std::vector<grammar::Token> tokens = g->stringToTokens(parseString);
  // inet::Node *output = inet::createParserNetwork(g->getStackActions(),
  // tokens);
  //
  // while (!inet::interactions.empty()) {
  //   // inet::drawNetwork(g, false);
  //   inet::interact();
  // }
  // inet::drawNetwork(g, false);
  //
  // std::cout << "Parsing results: " << std::endl;
  // std::vector<grammar::ParseTree *> trees = g->getParses(output);
  // for (grammar::ParseTree *tree : trees) {
  //   g->printParseTree(tree);
  //   std::cout << std::endl;
  // }
  //
  // std::cout << inet::total_interactions << std::endl;

  return 0;
};
