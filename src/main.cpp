#include <MappinGen/MappinGen.hpp>
#include <MappinUtil/MappinUtil.hpp>
#include <iostream>

class TestException : public MappinException {
  const char *message() const override { return "Test Error"; }

  TestException(const char *file, Span span, std::string line)
      : MappinException(file, span, line) {}

  friend class MappinException;
};

int main() {
  GrammarParser g_parser("examples/test.grammar");
  std::unique_ptr<grammar::Grammar> g = g_parser.parseGrammar();
  g->print();

  // MappinException *e =
  //     MappinException::newMappinException("examples/test.grammar",
  //                                         {{0, 5}, {0, 10}}, "Hello
  //                                         Exception!")
  //         .value();
  TestException *e =
      MappinException::newMappinException<TestException>(
          "examples/test.grammar", {{0, 5}, {0, 10}}, "Hello Exception!")
          .value();

  std::cerr << e->what() << std::endl;
  return 0;
};
