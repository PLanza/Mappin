#include "../include/generate/grammar/llgrammar.hpp"
#include "../include/generate/grammar/lrgrammar.hpp"
#include "../include/generate/grammar_parser.hpp"
#include "../include/generate/util.hpp"
#include "../include/gpu/run.hpp"
#include "../include/parse/draw.hpp"
#include "../include/parse/inet.hpp"
#include "../include/parse/nodes.hpp"
#include "../include/parse/parser.hpp"
#include <iostream>

int main() {
  std::unique_ptr<grammar::Grammar> g;
  try {
    GrammarParser<grammar::LRGrammar> g_parser("examples/test3.grammar");
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

  // std::string parseString = "B B B B B B B B B B B B X Z Z Z Z Z Z Z Z Z Z Z
  // Z "
  //                           "C C C C C C C C C C C C Y";
  // std::string parseString =
  //     "LCURL STRING COLON LCURL STRING COLON STRING COMMA STRING COLON LCURL
  //     " "STRING COLON STRING COMMA STRING COLON LCURL STRING COLON LCURL
  //     STRING " "COLON STRING COMMA STRING COLON STRING COMMA STRING COLON
  //     STRING COMMA " "STRING COLON LCURL STRING COLON STRING COMMA STRING
  //     COLON LSQUARE " "STRING COMMA STRING RSQUARE RCURL COMMA STRING COLON
  //     STRING RCURL RCURL " "RCURL RCURL RCURL";
  std::string parseString = "L S CL L R CM S CL L S CL S CM S CL L R R R";

  parse(std::move(g), parseString);
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
