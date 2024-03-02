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

enum ParseActionKind { EMPTY, SHIFT, REDUCE };

struct ParseAction {
  ParseActionKind kind;
  // TODO: change to optional
  uint32_t reduce_rule; // points to a rule in the grammar vector
};

struct StackAction {
  std::vector<Token> lhs;
  std::vector<Token> rhs;
};

class ParseTable {
public:
  virtual ParseAction getAction(uint32_t, uint32_t) = 0;
};

// line l: a -> A b c
// => (a, [A, b, c], l)
typedef std::vector<std::tuple<Token, std::vector<Token>, std::size_t>>
    grammar_rules;

enum GrammarExceptionKind {
  AMBIGUOUS_GRAMMAR,
};

MappinException *exceptionOnLine(GrammarExceptionKind, const char *,
                                 std::size_t);

class Grammar {
public:
  Token newToken(TokenKind, std::string);
  Token newTerminal(std::string);
  Token newNonTerminal(std::string);

  void addRule(std::string, std::vector<Token>, bool, std::size_t);
  virtual void makeParseTable() = 0;
  virtual std::vector<StackAction> *generateStackActions() = 0;

  void printGrammar();
  virtual void printParseTable() = 0;

protected:
  Grammar(const char *);
  void lockGrammar();

  const char *file_name;
  uint32_t start_rule;

  uint32_t terms_size;
  uint32_t nonterms_size;

  // TODO: change to unique pointers
  std::string *terminals;
  std::string *nonterminals;

  grammar_rules rules;
  // TODO: change to unique pointer
  ParseTable *parse_table;

  bool lock;

private:
  std::unordered_map<std::string, uint32_t> term_id_map;
  std::unordered_map<std::string, uint32_t> nonterm_id_map;
};

class LLParseTable : public ParseTable {
public:
  LLParseTable(uint32_t, uint32_t, grammar_rules &, const char *);
  ~LLParseTable();

  ParseAction getAction(uint32_t, uint32_t) override;

private:
  ParseAction *table;
  uint32_t rows;
  uint32_t cols;
};

class LLGrammar : public Grammar {
public:
  LLGrammar(const char *);
  void makeParseTable() override;
  std::vector<StackAction> *generateStackActions() override;
  void printParseTable() override;

private:
  std::vector<Token> findTerminal(uint32_t, uint32_t);
};

} // namespace grammar

#endif
