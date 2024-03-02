#ifndef __MAPPIN_GEN_GRAMMAR__
#define __MAPPIN_GEN_GRAMMAR__

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <util/util.hpp>

namespace grammar {

enum TokenKind { TERM, NON_TERM, ANY, REST };

struct Token {
  TokenKind kind;
  uint32_t id;
  std::string name;
};

const Token ANY_TOKEN = Token{ANY, 0, ""};
const Token REST_TOKEN = Token{REST, 0, ""};

enum ParseActionKind { EMPTY, SHIFT, REDUCE };

struct ParseAction {
  ParseActionKind kind;
  uint32_t reduce_rule; // points to a rule in the grammar vector
};

struct StackAction {
  std::vector<Token> lhs;
  std::vector<Token> rhs;
  std::vector<uint32_t> reduce_rules;
};

class ParseTable {
public:
  virtual ParseAction getAction(uint32_t, uint32_t) = 0;
};

enum GrammarExceptionKind {
  AMBIGUOUS_GRAMMAR,
};

MappinException *exceptionOnLine(GrammarExceptionKind, const char *,
                                 std::size_t);

// line l: a -> A b c
// => (a, [A, b, c], l)
typedef std::vector<std::tuple<Token, std::vector<Token>, std::size_t>>
    grammar_rules;

class Grammar {
public:
  Token newToken(TokenKind, std::string);

  void addRule(std::string, std::vector<Token>, bool, std::size_t);
  virtual void makeParseTable() = 0;
  virtual void generateStackActions() = 0;

  void printGrammar();
  void printToken(Token token);
  virtual void printParseTable() = 0;
  void printStackActions();

protected:
  Grammar(const char *);

  void fillStringArrays();

  const char *file_name;
  uint32_t start_rule;

  uint32_t terms_size;
  uint32_t nonterms_size;

  std::string *terminals;
  std::string *nonterminals;

  grammar_rules rules;
  ParseTable *parse_table;
  std::vector<StackAction> *stack_actions;

private:
  std::unordered_map<std::string, uint32_t> term_id_map;
  std::unordered_map<std::string, uint32_t> nonterm_id_map;

  Token newTerminal(std::string);
  Token newNonTerminal(std::string);
};

} // namespace grammar

#endif
