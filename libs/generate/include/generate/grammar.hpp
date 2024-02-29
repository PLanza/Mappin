#ifndef __MAPPIN_GEN_GRAMMAR__
#define __MAPPIN_GEN_GRAMMAR__

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace grammar {

// TODO: Associate each unique terminal/non-terminal with a unique ID
// This should speed up searches and comparisons

enum TokenKind { TERM, NON_TERM };

struct Token {
  TokenKind kind;
  unsigned int id;
  std::string name;
};

typedef std::vector<std::pair<Token, std::vector<Token>>> grammar_rules;

class Grammar {
public:
  Grammar();

  Token newToken(TokenKind kind, std::string name);
  Token newTerminal(std::string name);
  Token newNonTerminal(std::string name);

  void addRule(std::string, std::vector<Token>, bool);

  void print();

private:
  grammar_rules rules;
  unsigned int start_rule;

  unsigned int next_term_id;
  unsigned int next_nonterm_id;

  std::unordered_map<std::string, unsigned int> term_id_map;
  std::unordered_map<std::string, unsigned int> nonterm_id_map;
};

enum ParseActionKind { EMPTY, SHIFT, REDUCE };

struct ParseAction {
  ParseActionKind kind;
  unsigned int reduce_rule;
};

class LLGrammar {
public:
  LLGrammar(Grammar grammar);

private:
  ParseAction **makeParseTable(grammar_rules rules);
};

} // namespace grammar

#endif
