#ifndef __MAPPIN_GEN_GRAMMAR__
#define __MAPPIN_GEN_GRAMMAR__

// TODO:
// - refactor to a virtual ParseTable class for the different kinds of grammars
// - create parse table as rules are parsed in order to give positioned errors

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace grammar {

enum TokenKind { TERM, NON_TERM };

struct Token {
  TokenKind kind;
  unsigned int id;
  std::string name;
};

enum ParseActionKind { EMPTY, SHIFT, REDUCE };

struct ParseAction {
  ParseActionKind kind;
  unsigned int reduce_rule; // points to a rule in the grammar vector
};

typedef std::vector<std::pair<Token, std::vector<Token>>> grammar_rules;

class Grammar {
public:
  Token newToken(TokenKind kind, std::string name);
  Token newTerminal(std::string name);
  Token newNonTerminal(std::string name);

  void addRule(std::string, std::vector<Token>, bool);
  virtual ParseAction *makeParseTable();

  void print();

protected:
  Grammar();
  unsigned int next_term_id;
  unsigned int next_nonterm_id;
  grammar_rules rules;

private:
  unsigned int start_rule;

  std::unordered_map<std::string, unsigned int> term_id_map;
  std::unordered_map<std::string, unsigned int> nonterm_id_map;
};

class LLGrammar : public Grammar {
public:
  LLGrammar();
  ParseAction *makeParseTable();
};

} // namespace grammar

#endif
