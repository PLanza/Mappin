#ifndef __MAPPIN_GEN_GRAMMAR__
#define __MAPPIN_GEN_GRAMMAR__

// TODO:
// - refactor to a virtual ParseTable class for the different kinds of grammars
// - include span data in grammar

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <util/util.hpp>

namespace grammar {

enum TokenKind { TERM, NON_TERM };

struct Token {
  TokenKind kind;
  uint32_t id;
  std::string name;
};

enum ParseActionKind { EMPTY, SHIFT, REDUCE };

struct ParseAction {
  ParseActionKind kind;
  unsigned int reduce_rule; // points to a rule in the grammar vector
};

typedef std::vector<std::tuple<Token, std::vector<Token>, std::size_t>>
    grammar_rules;

enum GrammarExceptionKind { AMBIGUOUS_GRAMMAR, UNABLE_TO_OPEN_FILE };

class Grammar {
public:
  Token newToken(TokenKind kind, std::string name);
  Token newTerminal(std::string name);
  Token newNonTerminal(std::string name);

  void addRule(std::string, std::vector<Token>, bool, std::size_t);
  virtual ParseAction *makeParseTable();

  void print();

protected:
  Grammar();

  const char *file_name;

  uint32_t next_term_id;
  uint32_t next_nonterm_id;

  grammar_rules rules;

  MappinException *exceptionOnLine(GrammarExceptionKind kind, std::size_t line);

private:
  unsigned int start_rule;

  std::unordered_map<std::string, uint32_t> term_id_map;
  std::unordered_map<std::string, uint32_t> nonterm_id_map;
};

class LLGrammar : public Grammar {
public:
  LLGrammar();
  ParseAction *makeParseTable();
};

} // namespace grammar

#endif
