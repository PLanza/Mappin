#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

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
    : file_name(file_name), next_nonterm_id(0), next_term_id(2), rules({}),
      parse_table(nullptr), term_id_map({{"_START_", 0}, {"_END_", 1}}){};

Token Grammar::newToken(TokenKind kind, std::string name) {
  if (kind == TERM)
    return this->newTerminal(name);
  else
    return this->newNonTerminal(name);
}

Token Grammar::newTerminal(std::string name) {
  if (auto id_index = this->term_id_map.find(name);
      id_index == this->term_id_map.end()) {
    term_id_map[name] = this->next_term_id++;
  }
  return {TERM, term_id_map[name], name};
}

Token Grammar::newNonTerminal(std::string name) {
  if (auto id_index = this->nonterm_id_map.find(name);
      id_index == this->nonterm_id_map.end()) {
    nonterm_id_map[name] = this->next_nonterm_id++;
  }
  return {NON_TERM, nonterm_id_map[name], name};
}

void Grammar::addRule(std::string name, std::vector<Token> rhs, bool start,
                      std::size_t line) {
  Token head = this->newNonTerminal(name);

  if (start) {
    this->start_rule = head.id;
  }

  this->rules.push_back({head, rhs, line});
}

void Grammar::print() {
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
  case UNABLE_TO_OPEN_FILE:
    return MappinException::newMappinException<UnableToOpenFileException>(
               file_name, {{1, 1}, {1, 1}}, std::nullopt)
        .value();
  default:
    // Should never reach here but makes compiler happy
    return nullptr;
  }
}

LLGrammar::LLGrammar(const char *file_name) : Grammar(file_name) {}

void LLGrammar::makeParseTable() {
  this->parse_table = new LLParseTable(
      this->next_term_id, this->next_nonterm_id, this->rules, this->file_name);
  this->parse_table->print();
}

LLParseTable::LLParseTable(uint32_t terminals, uint32_t nonterminals,
                           grammar_rules &rules, const char *file_name)
    : rows(nonterminals), cols(terminals) {
  std::cout << "Making a " << this->rows << " x " << this->cols
            << " LL parse table" << std::endl;

  // Initialize parse table as a contiguous array
  ParseAction *parse_table = new ParseAction[this->rows * this->cols];
  for (int i = 0; i < this->rows; i++) {
    for (int j = 0; j < this->cols; j++) {
      parse_table[i * this->cols + j] = ParseAction{EMPTY, 0};
    }
  }

  // Initialize first set
  // first 32 bytes are the terminal id, last 32 bytes are the rule index
  std::unordered_set<uint64_t> first_set[this->rows];

  bool not_done;
  do {
    not_done = false;
    for (uint32_t i = 0; i < rules.size(); i++) {
      const auto &[head, rhs, _] = rules[i];
      Token first = rhs[0];

      if (first.kind == TERM)
        not_done |= first_set[head.id]
                        .insert(static_cast<uint64_t>(first.id) << 32 |
                                static_cast<uint64_t>(i))
                        .second;
      else {
        for (auto &element : first_set[first.id]) {
          not_done |= first_set[head.id]
                          .insert(element & (0xFFFFFFFFull << 32) |
                                  static_cast<uint64_t>(i))
                          .second;
        }
      }
    }
  } while (not_done);

  // Set REDUCE rules from first set
  for (uint32_t non_term = 0; non_term < this->rows; non_term++) {
    for (const uint64_t &element : first_set[non_term]) {
      uint32_t term = static_cast<uint32_t>(element >> 32);
      uint32_t rule = static_cast<uint32_t>(element);

      if (parse_table[non_term * this->cols + term].kind != EMPTY) {
        throw exceptionOnLine(AMBIGUOUS_GRAMMAR, file_name,
                              std::get<2>(rules[rule]));
      }
      parse_table[non_term * this->cols + term] = {REDUCE, rule};
    }
  }

  this->table = parse_table;
}

void LLParseTable::print() {
  for (int i = 0; i < this->rows; i++) {
    for (int j = 0; j < this->cols; j++) {
      switch (this->table[i * this->cols + j].kind) {
      case grammar::EMPTY: {
        std::cout << "   ";
        break;
      }
      case grammar::SHIFT: {
        std::cout << "S  ";
        break;
      }
      case grammar::REDUCE: {
        std::cout << "R" << this->table[i * 7 + j].reduce_rule << " ";
        break;
      }
      }
    }
    std::cout << std::endl;
  }
}

LLParseTable::~LLParseTable() { delete[] this->table; }

} // namespace grammar
