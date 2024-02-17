cmake_minimum_required(VERSION 3.18)

set(PROJECT_NAME
  MappinGen
)

set(LIBRARY_NAME
  MappinGen
)

set(LIBRARY_SOURCE_DIR
    src
)
set(LIBRARY_SOURCE
    ${LIBRARY_SOURCE_DIR}/MappinGen.cpp
)

set(LIBRARY_HEADERS_DIR
  include/${LIBRARY_NAME}
)
set(LIBRARY_HEADERS
    # ${LIBRARY_HEADER_DIR}/MappinGen.h
)

project(${PROJECT_NAME}
        DESCRIPTION "Mappin's parser generator"
        LANGUAGES CXX
)

add_library(${LIBRARY_NAME} STATIC ${LIBRARY_HEADERS} ${LIBRARY_SOURCE})

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
