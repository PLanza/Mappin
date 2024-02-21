#include <cctype>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "MappinGen.hpp"

class UnableToOpenFileException : public MappinException {
  const char *message() const override {
    std::string msg = "Unable to open file ";
    msg += this->file;
    msg.push_back('.');

    char *msg_c_str = new char[msg.length() + 1];
    strcpy(msg_c_str, msg.c_str());

    return msg_c_str;
  }

  UnableToOpenFileException(const char *file, Span span, std::string line)
      : MappinException(file, span, line) {}

  friend class MappinException;
};

class ExpectedNonTerminalException : public MappinException {
  const char *message() const override {
    return "Expected a lowercase non-terminal on the left hand side of a "
           "grammar rule.";
  }

  ExpectedNonTerminalException(const char *file, Span span, std::string line)
      : MappinException(file, span, line) {}

  friend class MappinException;
};

class ExpectedTermNonTermException : public MappinException {
  const char *message() const override {
    return "Expected either a terminal or non-terminal on the right hand side "
           "of a grammar rule.";
  }

  ExpectedTermNonTermException(const char *file, Span span, std::string line)
      : MappinException(file, span, line) {}

  friend class MappinException;
};

GrammarParser::GrammarParser(const char *file_name)
    : file_name(file_name), pos({1, 0}), line_offset(0),
      grammar(new grammar::Grammar) {
  this->grammar_fs.open(this->file_name);

  if (this->grammar_fs.is_open())
    std::getline(this->grammar_fs, this->curr_line);
  else
    std::cout << "Couldn't open file" << std::endl;
};

bool GrammarParser::eof() { return this->grammar_fs.eof(); }
bool GrammarParser::endOfLine() {
  return this->curr_line.empty() ||
         this->line_offset >= this->curr_line.length() - 1;
}

// Get the character at the parser head
char GrammarParser::getChar() {
  return (this->curr_line.empty()) ? '\n' : this->curr_line[line_offset];
}

// Bump the parser head to the next character
// If the EOF is reached, then `false` is returned
bool GrammarParser::bump() {
  if (this->eof())
    return false;
  if (this->endOfLine()) {
    this->line_offset = 0;
    this->pos.column = 1;
    this->pos.line += 1;

    std::getline(this->grammar_fs, this->curr_line);
  } else {
    this->line_offset += 1;
    this->pos.column += 1;
  }

  return this->line_offset < this->curr_line.length();
}

// Bumps the parser head until a non-space character is reached
// If a new line is bumber, then `true` is returned
bool GrammarParser::bumpSpace() {
  bool new_line = false;

  while (!this->eof()) {
    if (this->curr_line.empty()) {
      this->bump();
      new_line = true;
    } else if (isspace(this->getChar())) {
      this->bump();
    } else
      break;
  }

  return new_line;
}

// Bumps the parser and also any subsequent space
// If the EOF is reached, the `false` is returned
bool GrammarParser::bumpAndBumpSpace() {
  if (!this->bump())
    return false;

  this->bumpSpace();
  return !this->eof();
}

// Bump the parser head past `prefix``, if parser head points to `prefix`
// If parser head does not point to `prefix` then return false
bool GrammarParser::bumpIf(const char *prefix) {
  if (this->curr_line.find(prefix, this->line_offset) == this->line_offset) {
    for (int i = 0; i < strlen(prefix); i++) {
      this->bump();
    }
    return true;
  } else
    return false;
}

MappinException *GrammarParser::exceptionAtSpan(GrammarParserExceptionKind kind,
                                                Span span) {
  switch (kind) {
  case EXPECTED_NON_TERM:
    return MappinException::newMappinException<ExpectedNonTerminalException>(
               this->file_name, span, this->curr_line)
        .value();
  case EXPECTED_TERM_NON_TERM:
    return MappinException::newMappinException<ExpectedTermNonTermException>(
               this->file_name, span, this->curr_line)
        .value();
  default:
    return MappinException::newMappinException<MappinException>(
               this->file_name, span, this->curr_line)
        .value();
  }
}

MappinException *
GrammarParser::exceptionAtParserHead(GrammarParserExceptionKind kind) {
  return this->exceptionAtSpan(kind, {this->pos, this->pos});
}

MappinException *
GrammarParser::exceptionFromLineStart(GrammarParserExceptionKind kind) {
  return this->exceptionAtSpan(kind, {{this->pos.line, 1}, this->pos});
}

std::unique_ptr<grammar::Grammar> GrammarParser::parseGrammar() {
  std::cout << "Parsing Grammar file " << this->file_name << std::endl;

  while (!this->eof()) {
    this->parseGrammarDefinition();
  }

  std::cout << "Done!" << std::endl;
  return std::move(this->grammar);
}

// Parses an instance of a grammar definition. For example:
//  a := b c
//     | A
void GrammarParser::parseGrammarDefinition() {
  std::string def_name = "";
  bool start_rule = false;

  if (this->getChar() == '$') {
    this->bumpAndBumpSpace();
    start_rule = true;
  }

  while (!this->eof() && (islower(this->getChar()) || this->getChar() == '_')) {
    def_name.push_back(this->getChar());
    this->bump();
  }
  this->bumpSpace();

  if (!this->bumpIf(":="))
    throw this->exceptionFromLineStart(EXPECTED_NON_TERM);

  this->bumpSpace();

  std::vector<std::vector<grammar::TermOrNonTerm>> right_hand_sides = {
      this->parseGrammarRHS()};
  while (this->getChar() == '|') {
    this->bumpAndBumpSpace();
    right_hand_sides.push_back(this->parseGrammarRHS());
  }

  this->grammar->addRule(def_name, right_hand_sides, start_rule);
}

// Parses the right hand side of a grammar definition.
// E.g. `b c` in `a := b c`
std::vector<grammar::TermOrNonTerm> GrammarParser::parseGrammarRHS() {
  std::vector<grammar::TermOrNonTerm> right_hand_side;

  uint32_t line = this->pos.line;
  while (line == this->pos.line) {
    grammar::TermKind kind;
    if (islower(this->getChar()))
      kind = grammar::NON_TERM;
    else if (isupper(this->getChar()))
      kind = grammar::TERM;
    else
      throw this->exceptionAtParserHead(EXPECTED_TERM_NON_TERM);

    std::string name;
    while (isalpha(this->getChar()) || this->getChar() == '_') {
      name.push_back(this->getChar());
      this->bump();

      if (line != this->pos.line)
        // Reached end of line so stop parsing
        break;
    }
    this->bumpSpace();
    char *name_array = new char[name.length() + 1];
    strcpy(name_array, name.c_str());
    right_hand_side.push_back(kind == grammar::TERM
                                  ? grammar::newTerminal(name_array)
                                  : grammar::newNonTerminal(name_array));
  }

  return right_hand_side;
}
