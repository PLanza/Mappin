#ifndef __MAPPIN_GEN__
#define __MAPPIN_GEN__

#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "Grammar.hpp"
#include <MappinUtil/MappinUtil.hpp>

enum GrammarParserExceptionKind {
  UNABLE_TO_OPEN_FILE,
  EXPECTED_NON_TERM,
  EXPECTED_TERM_NON_TERM
};

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

  MappinException *exceptionAtSpan(GrammarParserExceptionKind kind, Span span);
  MappinException *exceptionAtParserHead(GrammarParserExceptionKind kind);
  MappinException *exceptionFromLineStart(GrammarParserExceptionKind kind);

public:
  GrammarParser(const char *file_name);
  std::unique_ptr<grammar::Grammar> parseGrammar();
  void parseGrammarDefinition();
  std::vector<grammar::TermOrNonTerm> parseGrammarRHS();
};

#endif
