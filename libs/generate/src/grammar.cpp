#include <fstream>
#include <iostream>

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
    : file_name(file_name), rules({}), parse_table(nullptr),
      stack_actions(nullptr), lock(false), nonterms_size(0), terminals(nullptr),
      nonterminals(nullptr), nonterm_id_map({}) {
  this->terms_size = 0;
  this->term_id_map = {};

  this->newTerminal("<");
  this->newTerminal(">");
}

Token Grammar::newToken(TokenKind kind, std::string name) {
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
  return {TERM, term_id_map[name], name};
}

Token Grammar::newNonTerminal(std::string name) {
  if (auto id_index = this->nonterm_id_map.find(name);
      id_index == this->nonterm_id_map.end()) {
    nonterm_id_map[name] = this->nonterms_size++;
  }
  return {NON_TERM, nonterm_id_map[name], name};
}

void Grammar::addRule(std::string name, std::vector<Token> rhs, bool start,
                      std::size_t line) {
  if (this->lock) {
    std::cerr << "ERROR: Attempted to add rule to grammar after it was locked."
              << std::endl;
    return;
  }

  Token head = this->newNonTerminal(name);

  if (start)
    this->start_rule = rules.size();

  this->rules.push_back({head, rhs, line});
}

void Grammar::printGrammar() {
  for (const auto &[head, rhs, _] : this->rules) {
    if (head.id == this->start_rule)
      std::cout << "$ ";
    std::cout << head.name << " := ";

    for (auto &token : rhs) {
      std::cout << "(" << token.id << ": " << token.name << ") ";
    }
    std::cout << "\n";
  }
  std::cout << std::endl;
}

void Grammar::printToken(Token token) {
  switch (token.kind) {
  case TERM: {
    std::cout << this->terminals[token.id];
    break;
  }
  case NON_TERM: {
    std::cout << this->nonterminals[token.id];
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

void Grammar::printStackActions() {
  if (!this->stack_actions) {
    std::cerr
        << "ERROR: Attempted to print stack actions before they were generated"
        << std::endl;
    return;
  }

  for (uint32_t i = 0; i < this->terms_size; i++) {
    printf("%8s : ", this->terminals[i].c_str());

    for (const auto &stack_action : stack_actions[i]) {
      for (const grammar::Token &token : stack_action.lhs) {
        this->printToken(token);
      }
      std::cout << "/";
      for (const grammar::Token &token : stack_action.rhs) {
        this->printToken(token);
      }
      std::cout << "  ";
    }
    std::cout << std::endl;
  }
}

void Grammar::lockGrammar() {
  this->lock = true;

  this->terminals = new std::string[this->terms_size];
  for (const auto &[name, t_id] : this->term_id_map)
    this->terminals[t_id] = name;

  this->nonterminals = new std::string[this->nonterms_size];
  for (const auto &[name, nt_id] : this->nonterm_id_map)
    this->nonterminals[nt_id] = name;
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
