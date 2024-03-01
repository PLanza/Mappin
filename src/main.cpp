#include <generate/grammar_parser.hpp>
#include <iostream>
#include <util/util.hpp>

int main() {
  try {
    GrammarParser g_parser("examples/test.grammar");
    std::unique_ptr<grammar::Grammar> g = g_parser.parseGrammar();
    g->print();
    grammar::ParseAction *parse_table = g->makeParseTable();
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 7; j++) {
        switch (parse_table[i * 7 + j].kind) {
        case grammar::EMPTY: {
          std::cout << "   ";
          break;
        }
        case grammar::SHIFT: {
          std::cout << "S  ";
          break;
        }
        case grammar::REDUCE: {
          std::cout << "R" << parse_table[i * 7 + j].reduce_rule << " ";
          break;
        }
        }
      }
      std::cout << std::endl;
    }
  } catch (MappinException *e) {
    std::cerr << e->what() << std::endl;
  }
  return 0;
};
