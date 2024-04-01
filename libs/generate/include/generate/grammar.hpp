#ifndef __MAPPIN_GEN_GRAMMAR__
#define __MAPPIN_GEN_GRAMMAR__

#include <boost/container_hash/extensions.hpp>
#include <boost/container_hash/hash_fwd.hpp>
#include <cstdint>
#include <deque>
#include <string>
#include <unordered_map>
#include <vector>

#include <parse/inet.hpp>
#include <util/util.hpp>

namespace grammar {

enum TokenKind { TERM, NON_TERM };

enum StackStateKind { SOME, ANY, REST };
struct StackState {
  StackStateKind kind;
  uint32_t value;

  bool operator==(grammar::StackState const &other) const {
    return this->value == other.value && this->kind == other.kind;
  }
};

const StackState REST_STATE = {REST, 0};
const StackState ANY_STATE = {ANY, 0};

struct StackAction {
  std::deque<StackState> lhs;
  std::deque<StackState> rhs;
  std::deque<uint32_t> reduce_rules;

  bool operator==(grammar::StackAction const &other) const {
    return this->lhs == other.lhs && this->rhs == other.rhs &&
           this->reduce_rules == other.reduce_rules;
  }
};

struct Token {
  TokenKind kind;
  uint32_t id;

  StackState toStackState() {
    if (this->kind == TERM) {
      return StackState{SOME, this->id * 2};
    } else {
      return StackState{SOME, this->id * 2 + 1};
    }
  }

  bool operator==(grammar::Token const &other) const {
    return this->id == other.id && this->kind == other.kind;
  }
};

enum ParseActionKind { EMPTY, SHIFT, REDUCE, ACCEPT };

struct ParseAction {
  ParseActionKind kind;
  uint32_t reduce_rule; // points to a rule in the grammar vector
};

class ParseTable {
public:
  virtual ParseAction getAction(uint32_t, uint32_t) const = 0;
  virtual ~ParseTable() = default;
};

enum GrammarExceptionKind {
  AMBIGUOUS_GRAMMAR,
};

MappinException *exceptionOnLine(GrammarExceptionKind, const char *,
                                 std::size_t);

struct ParseTree {
  TokenKind kind;
  uint32_t value;
  ParseTree **children = nullptr;

  ParseTree(TokenKind kind, uint32_t value, size_t size)
      : kind(kind), value(value) {
    this->children = new ParseTree *[size];
  }
  ~ParseTree() { delete[] children; }
};

// line l: a -> A b c
// => (a, [A, b, c], l)
typedef std::vector<std::tuple<Token, std::vector<Token>, std::size_t>>
    grammar_rules;

class Grammar {
public:
  Token newToken(TokenKind, std::string);

  void addRule(std::string, std::vector<Token>, bool, std::size_t);
  virtual void finalize() = 0;
  virtual void makeParseTable() = 0;
  virtual void generateStackActions() = 0;

  std::vector<StackAction> *getStackActions();
  std::tuple<Token, std::vector<Token>, std::size_t> &getRule(size_t);
  std::string &getTerminalString(uint32_t);
  std::string &getNonTerminalString(uint32_t);

  std::vector<Token> stringToTokens(std::string);

  std::vector<ParseTree *> getParses(inet::Node *);

  void printGrammar();
  void printStackState(StackState, bool);
  virtual void printParseTable() = 0;
  void printStackAction(const StackAction &, bool);
  void printStackActions(bool);
  virtual void printStackActions() = 0;
  virtual void printParseTree(ParseTree *) = 0;

  virtual ~Grammar();

protected:
  Grammar(const char *);

  const char *file_name;
  uint32_t start_rule;

  uint32_t terms_size;
  uint32_t nonterms_size;

  std::string *terminals = nullptr;
  std::string *nonterminals = nullptr;

  grammar_rules rules;
  std::vector<StackAction> *stack_actions = nullptr;

  std::unordered_map<std::string, uint32_t> term_id_map;
  std::unordered_map<std::string, uint32_t> nonterm_id_map;

  Token newTerminal(std::string);
  Token newNonTerminal(std::string);
  void fillStringArrays();
  virtual ParseTable *getParseTable() = 0;
  virtual ParseTree *getParse(inet::Node *) = 0;
};

} // namespace grammar

namespace boost {
template <> struct hash<grammar::Token> {
  size_t operator()(const grammar::Token &t) const {
    size_t hash = 0;
    boost::hash_combine(hash, t.kind);
    boost::hash_combine(hash, t.id);
    return hash;
  }
};
} // namespace boost

#endif
