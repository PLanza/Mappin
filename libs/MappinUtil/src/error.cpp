#include "MappinUtil.hpp"
#include <algorithm>
#include <cstddef>
#include <cstring>

MappinException::MappinException(const char *file, Span span, std::string line)
    : file(file), span(span), line(line) {}

// template <class T>
// std::optional<T *> MappinException::newMappinException(const char *file,
//                                                        Span span,
//                                                        std::string line) {
//   if (span.start.line != span.end.line && span.start.column >
//   span.end.column)
//     return std::nullopt;
//
//   return new T(file, span, line);
// }

const char *MappinException::what() const noexcept {
  std::string err_str = "error: ";

  err_str += this->message();
  err_str += "\n  --> ";
  err_str += this->file;
  err_str += "\n\n";

  std::string line_str = std::to_string(this->span.start.line);
  size_t line_digits = line_str.length();

  line_str += ": ";
  line_str += this->line;
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

  char *err_c_str = new char[err_str.length() + 1];
  strcpy(err_c_str, err_str.c_str());

  return err_c_str;
}

const char *MappinException::message() const { return "Mappin Exception"; }
