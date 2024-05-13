#include "../../include/generate/util.hpp"
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <optional>

MappinException::MappinException(const char *file, Span span, std::string line)
    : file(file), span(span),
      line(line == "" ? std::nullopt : std::optional<std::string>(line)) {}

const char *MappinException::what() const noexcept {
  std::string err_str = "error: ";

  err_str += this->message();
  err_str += "\n  --> ";
  err_str += this->file;
  err_str += "\n";

  if (this->line.has_value()) {
    err_str += "\n";
    std::string line_str = std::to_string(this->span.start.line);
    size_t line_digits = line_str.length();

    line_str += ": ";
    line_str += this->line.value();
    line_str += "\n";

    for (int i = 0; i < line_digits + 1 + this->span.start.column; i++) {
      line_str.push_back(' ');
    }
    unsigned int span_width = span.end.column - span.start.column;
    for (int i = 0; i < std::max(1u, span_width); i++) {
      line_str.push_back('^');
    }
    line_str += "\n";

    err_str += line_str;
  }

  char *err_c_str = new char[err_str.length() + 1];
  strcpy(err_c_str, err_str.c_str());

  return err_c_str;
}

const char *MappinException::message() const { return "Mappin Exception"; }

const char *UnableToOpenFileException::message() const {
  return "Unable to open file.";
}

UnableToOpenFileException::UnableToOpenFileException(const char *file,
                                                     Span span,
                                                     std::string line)
    : MappinException(file, span, line) {}
