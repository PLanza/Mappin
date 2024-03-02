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

enum ParseActionKind { EMPTY, SHIFT, REDUCE };

struct ParseAction {
  ParseActionKind kind;
  unsigned int reduce_rule; // points to a rule in the grammar vector
};

struct StackAction {
  std::vector<Token> lhs;
  std::vector<Token> rhs;
};

class ParseTable {
public:
  // virtual StackAction *generateStackActions() = 0;
  virtual void print() = 0;
};

typedef std::vector<std::tuple<Token, std::vector<Token>, std::size_t>>
    grammar_rules;

enum GrammarExceptionKind { AMBIGUOUS_GRAMMAR, UNABLE_TO_OPEN_FILE };

MappinException *exceptionOnLine(GrammarExceptionKind, const char *,
                                 std::size_t);

class Grammar {
public:
  Token newToken(TokenKind, std::string);
  Token newTerminal(std::string);
  Token newNonTerminal(std::string);

  void addRule(std::string, std::vector<Token>, bool, std::size_t);
  virtual void makeParseTable() = 0;

  void print();

protected:
  Grammar(const char *);

  const char *file_name;

  uint32_t next_term_id;
  uint32_t next_nonterm_id;

  grammar_rules rules;
  ParseTable *parse_table;

private:
  unsigned int start_rule;

  std::unordered_map<std::string, uint32_t> term_id_map;
  std::unordered_map<std::string, uint32_t> nonterm_id_map;
};

class LLParseTable : public ParseTable {
public:
  LLParseTable(uint32_t, uint32_t, grammar_rules &, const char *);
  ~LLParseTable();

  // StackAction *generateStackActions() override;
  void print() override;

private:
  ParseAction *table;
  uint32_t rows;
  uint32_t cols;
};

class LLGrammar : public Grammar {
public:
  LLGrammar(const char *);
  void makeParseTable() override;
};

} // namespace grammar

#endif
