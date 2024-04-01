#ifndef __MAPPIN_PARSE_PARSER__
#define __MAPPIN_PARSE_PARSER__

#include "inet.hpp"
#include <generate/grammar.hpp>

namespace inet {

Node *createParserNetwork(std::vector<grammar::StackAction> *,
                          std::vector<grammar::Token>);

} // namespace inet
#endif
