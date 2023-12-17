cmake_minimum_required(VERSION 3.10)

project(whisper_for_media_podium VERSION 0.0.1 LANGUAGES C CXX)

set(CMAKE_OSX_ARCHITECTURES "x86_64;arm64")

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC -pthread")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fPIC -pthread")

set(TARGET media_podium_whisper)

if(BUILD_EXEC)
  add_executable(${TARGET} main.cpp)
else()
  add_library(${TARGET} SHARED main.cpp)
endif()

set(WHISPER_BUILD_EXAMPLES ON)
add_subdirectory(whisper.cpp)

target_compile_definitions(${TARGET} PUBLIC DART_SHARED_LIB)

target_link_libraries(${TARGET} PRIVATE common whisper ${CMAKE_THREAD_LIBS_INIT})

