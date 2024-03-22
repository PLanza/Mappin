#ifndef __MAPPIN_PARSE_PARSER__
#define __MAPPIN_PARSE_PARSER__

#include <generate/grammar.hpp>

namespace inet{

void create_parser_network(
  std::vector<grammar::StackAction>* &, 
  std::vector<grammar::Token>);

}
#endif
