#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "grammar.hpp"
#include "util/util.hpp"

namespace grammar {

class AmbiguousGrammarException : public MappinException {
  const char *message() const override {
    return "The following rule makes the grammar ambiguous: ";
  }

  AmbiguousGrammarException(const char *file, Span span, std::string line)
      : MappinException(file, span, line) {}

  friend class MappinException;
};

Grammar::Grammar(const char *file_name)
    : file_name(file_name), rules({}), stack_actions(nullptr), nonterms_size(0),
      terminals(nullptr), nonterminals(nullptr), nonterm_id_map({}) {
  this->terms_size = 0;
  this->term_id_map = {};

  this->newTerminal("<");
  this->newTerminal(">");
}

Token Grammar::newToken(TokenKind kind, std::string name) {
  if (this->getParseTable() != nullptr) {
    std::cerr << "ERROR: Attempted to add token to grammar after it was locked."
              << std::endl;
  }

  if (kind == TERM)
    return this->newTerminal(name);
  else
    return this->newNonTerminal(name);
}

Token Grammar::newTerminal(std::string name) {
  if (auto id_index = this->term_id_map.find(name);
      id_index == this->term_id_map.end()) {
    term_id_map[name] = this->terms_size++;
  }
  return {TERM, term_id_map[name]};
}

Token Grammar::newNonTerminal(std::string name) {
  if (auto id_index = this->nonterm_id_map.find(name);
      id_index == this->nonterm_id_map.end()) {
    nonterm_id_map[name] = this->nonterms_size++;
  }
  return {NON_TERM, nonterm_id_map[name]};
}

void Grammar::addRule(std::string name, std::vector<Token> rhs, bool start,
                      std::size_t line) {
  if (this->getParseTable() != nullptr) {
    std::cerr << "ERROR: Attempted to add rule to grammar after it was locked."
              << std::endl;
    return;
  }

  Token head = this->newNonTerminal(name);

  if (start)
    this->start_rule = rules.size();

  this->rules.push_back({head, rhs, line});
}

std::vector<StackAction> *Grammar::getStackActions() {
  return this->stack_actions;
}

std::tuple<Token, std::vector<Token>, std::size_t> &
Grammar::getRule(size_t rule) {
  return this->rules[rule];
}
std::string &Grammar::getTerminalString(uint32_t id) {
  return this->terminals[id];
}
std::string &Grammar::getNonTerminalString(uint32_t id) {
  return this->nonterminals[id];
}

std::vector<Token> Grammar::stringToTokens(std::string input) {
  std::vector<Token> tokens = {{TERM, this->term_id_map["<"]}};

  std::stringstream input_stream = std::stringstream(input);
  std::string token_string;

  while (std::getline(input_stream, token_string, ' ')) {
    tokens.push_back({TERM, this->term_id_map[token_string]});
  }
  tokens.push_back({TERM, this->term_id_map[">"]});

  return tokens;
}

void Grammar::printGrammar() {
  for (uint32_t i = 0; i < this->rules.size(); i++) {
    const auto &[head, rhs, _] = this->rules[i];

    if (i == this->start_rule)
      std::cout << "$ ";

    std::cout << this->nonterminals[head.id] << " := ";

    for (auto &token : rhs) {
      std::cout << "(" << token.id << ": ";
      if (token.kind == TERM)
        std::cout << this->terminals[token.id];
      else
        std::cout << this->nonterminals[token.id];
      std::cout << ") ";
    }
    std::cout << "\n";
  }
  std::cout << std::endl;
}

void Grammar::printStackState(StackState state, bool as_token) {
  switch (state.kind) {
  case SOME: {
    if (!as_token) {
      std::cout << state.value;
      break;
    }
    if (state.value % 2 == 0)
      std::cout << this->terminals[state.value / 2];
    else
      std::cout << this->nonterminals[(state.value - 1) / 2];
    break;
  }
  case ANY: {
    std::cout << "_";
    break;
  }
  case REST: {
    std::cout << "-";
    break;
  }
  }
}

void Grammar::printStackAction(const StackAction &stack_action,
                               bool as_tokens) {
  for (const grammar::StackState &state : stack_action.lhs) {
    this->printStackState(state, as_tokens);
  }
  std::cout << "/";
  for (const grammar::StackState &state : stack_action.rhs) {
    this->printStackState(state, as_tokens);
  }

  if (stack_action.reduce_rules.empty())
    return;

  std::cout << " (";
  for (int j = 0; j < stack_action.reduce_rules.size(); j++) {
    std::cout << stack_action.reduce_rules[j];
    if (j < stack_action.reduce_rules.size() - 1)
      std::cout << ",";
  }
  std::cout << ")";
}

void Grammar::printStackActions(bool as_tokens) {
  if (this->stack_actions == nullptr) {
    std::cerr
        << "ERROR: Attempted to print stack actions before they were generated"
        << std::endl;
    return;
  }

  for (uint32_t i = 0; i < this->terms_size; i++) {
    printf("%8s : ", this->terminals[i].c_str());

    for (const auto &stack_action : stack_actions[i]) {
      this->printStackAction(stack_action, as_tokens);
      std::cout << " | ";
    }
    std::cout << std::endl;
  }
}

void Grammar::fillStringArrays() {
  this->terminals = new std::string[this->terms_size];
  for (const auto &[name, t_id] : this->term_id_map)
    this->terminals[t_id] = name;

  this->nonterminals = new std::string[this->nonterms_size];
  for (const auto &[name, nt_id] : this->nonterm_id_map)
    this->nonterminals[nt_id] = name;
}

Grammar::~Grammar() {
  delete[] this->terminals;
  delete[] this->nonterminals;
}

MappinException *exceptionOnLine(GrammarExceptionKind kind,
                                 const char *file_name, std::size_t line) {
  std::ifstream file_stream;
  file_stream.open(file_name);

  std::string grammar_line;

  for (int i = 1; i <= line; i++)
    std::getline(file_stream, grammar_line);

  Span span = Span{{line, 1}, {line, grammar_line.length() + 1}};

  switch (kind) {
  case AMBIGUOUS_GRAMMAR:
    return MappinException::newMappinException<AmbiguousGrammarException>(
               file_name, span, std::optional<std::string>(grammar_line))
        .value();
  default:
    // Should never reach here but makes compiler happy
    return nullptr;
  }
}

} // namespace grammar
