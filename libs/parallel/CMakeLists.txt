cmake_minimum_required(VERSION 3.18)

set(PROJECT_NAME
  parallel
)

set(LIBRARY_NAME
  parallel
)

set(SOURCE_DIR
    src
)
set(SOURCE
    ${SOURCE_DIR}/inet.cu
    ${SOURCE_DIR}/actions.cu
    ${SOURCE_DIR}/network.cu
    ${SOURCE_DIR}/run.cu
    ${SOURCE_DIR}/queue.cu
)

set(HEADERS_DIR
  include/${LIBRARY_NAME}
)
set(HEADERS
  ${HEADERS_DIR}/inet.cuh
  ${HEADERS_DIR}/actions.cuh
  ${HEADERS_DIR}/network.cuh
  ${HEADERS_DIR}/queue.cuh
)

set(LIBRARIES
  generate
)

project(${PROJECT_NAME}
        DESCRIPTION "The parallel implementation of Mappin's parser"
        LANGUAGES CXX CUDA
)

add_library(${LIBRARY_NAME} STATIC ${HEADERS} ${SOURCE})

target_link_libraries(${PROJECT_NAME} ${LIBRARIES})

target_include_directories(${LIBRARY_NAME} PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${LIBRARY_NAME}>
  $<INSTALL_INTERFACE:include/${LIBRARY_NAME}>
)

target_include_directories(${LIBRARY_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
