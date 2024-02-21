#ifndef __MAPPIN_UTILS__
#define __MAPPIN_UTILS__

#include <exception>
#include <optional>
#include <string>

struct Position {
  // Line number starting at `1`
  unsigned int line;
  // Column number starting at `1`
  unsigned int column;
  // Byte-offset from the start of the file, starting at `0`
  // uint32_t offset;
};

struct Span {
  Position start;
  Position end;
};

class MappinException : public std::exception {
private:
  Span span;
  std::optional<std::string> line;
  virtual const char *message() const;

protected:
  MappinException(const char *file, Span span, std::string line);
  const char *file;

public:
  const char *what() const noexcept override;

  template <class T>
  static std::optional<T *>
  newMappinException(const char *file, Span span,
                     std::optional<std::string> line) {
    if (span.start.line != span.end.line && span.start.column > span.end.column)
      return std::nullopt;

    if (line.has_value())
      return new T(file, span, line.value());
    else
      return new T(file, span, "");
  }
};

#endif
