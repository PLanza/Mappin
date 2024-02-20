#ifndef __MAPPIN_GEN_UTILS__
#define __MAPPIN_GEN_UTILS__

#include <cstdint>

struct Position {
  // Line number starting at `1`
  uint32_t line;
  // Column number starting at `1`
  uint32_t column;
  // Byte-offset from the start of the file, starting at `0`
  // uint32_t offset;
};

#endif
