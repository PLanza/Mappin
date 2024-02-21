cmake_minimum_required(VERSION 3.18)

set(PROJECT_NAME
  MappinGen
)

set(LIBRARY_NAME
  MappinGen
)

set(SOURCE_DIR
    src
)
set(SOURCE
    ${SOURCE_DIR}/MappinGen.cpp
    ${SOURCE_DIR}/ParseGrammar.cpp
    ${SOURCE_DIR}/Grammar.cpp
)

set(HEADERS_DIR
  include/${LIBRARY_NAME}
)
set(HEADERS
  ${HEADERS_DIR}/MappinGen.hpp
  ${HEADERS_DIR}/Grammar.hpp
)

set(LIBRARIES
  MappinUtil
)

project(${PROJECT_NAME}
        DESCRIPTION "Mappin's parser generator"
        LANGUAGES CXX
)

# foreach(LIBRARY ${LIBRARIES})
#   add_subdirectory("../${LIBRARY}" "../../build/libs/${LIBRARY}")
# endforeach(LIBRARY)

add_library(${LIBRARY_NAME} STATIC ${HEADERS} ${SOURCE})

target_link_libraries(${PROJECT_NAME} ${LIBRARIES})

# Imports MappinGen.h as `#include "MappinGen.h` when used internally
# and as `#include <MappinGen/MappinGen.h>` when used externally
target_include_directories(${LIBRARY_NAME} PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${LIBRARY_NAME}>
  $<INSTALL_INTERFACE:include/${LIBRARY_NAME}>
)

target_include_directories(${LIBRARY_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

