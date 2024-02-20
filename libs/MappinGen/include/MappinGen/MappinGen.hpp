#ifndef __MAPPIN_GEN__
#define __MAPPIN_GEN__

#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "Grammar.hpp"
#include "Utils.hpp"

void test(int x);

class GrammarParser {
private:
  const char *file_name;
  std::ifstream grammar_fs;
  Position pos;
  std::string curr_line;
  size_t line_offset;
  std::unique_ptr<grammar::Grammar> grammar;

  bool eof();
  bool endOfLine();
  char getChar();
  bool bump();
  bool bumpSpace();
  bool bumpAndBumpSpace();
  bool bumpIf(const char *prefix);

public:
  GrammarParser(const char *file_name);
  std::unique_ptr<grammar::Grammar> parseGrammar();
  void parseGrammarDefinition();
  std::vector<grammar::TermOrNonTerm> parseGrammarRHS();
};

#endif
