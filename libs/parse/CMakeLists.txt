cmake_minimum_required(VERSION 3.18)

set(PROJECT_NAME
  parse
)

set(LIBRARY_NAME
  parse
)

set(LIBRARY_SOURCE_DIR
    src
)
set(LIBRARY_SOURCE
    ${LIBRARY_SOURCE_DIR}/parse.cpp
)

set(LIBRARY_HEADERS_DIR
  include/${LIBRARY_NAME}
)
set(LIBRARY_HEADERS)

project(${PROJECT_NAME}
        DESCRIPTION "Mappin's parser"
        LANGUAGES CXX
)

add_library(${LIBRARY_NAME} STATIC ${LIBRARY_HEADERS} ${LIBRARY_SOURCE})

target_include_directories(${LIBRARY_NAME} PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${LIBRARY_NAME}>
  $<INSTALL_INTERFACE:include/${LIBRARY_NAME}>
)

target_include_directories(${LIBRARY_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
