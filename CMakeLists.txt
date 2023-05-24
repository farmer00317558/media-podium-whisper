cmake_minimum_required(VERSION 3.10)

project(whisper_library VERSION 0.0.1 LANGUAGES C CXX)
set(CMAKE_OSX_ARCHITECTURES "x86_64;arm64")
# project(whisper_executable VERSION 0.0.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC -pthread")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fPIC -pthread")

add_library(whisper SHARED main.cpp)
# add_executable(whisper main.cpp)
add_library(whisper_lib whisper.cpp/whisper.cpp)
add_library(ggml whisper.cpp/ggml.c)

set_target_properties(whisper PROPERTIES
  PUBLIC_HEADER whisper.cpp/whisper.h 
  OUTPUT_NAME "whisper"
)

if (WIN32)
  set_target_properties(whisper PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
  )
endif (WIN32)


# on APPLE
if (APPLE)
    # include Accelerate framework
    if (NOT WHISPER_NO_ACCELERATE)
        find_library(ACCELERATE_FRAMEWORK Accelerate)

        if (ACCELERATE_FRAMEWORK)
            message(STATUS "Accelerate framework found")

            set(WHISPER_EXTRA_LIBS  ${WHISPER_EXTRA_LIBS}  ${ACCELERATE_FRAMEWORK})
            set(WHISPER_EXTRA_FLAGS ${WHISPER_EXTRA_FLAGS} -DGGML_USE_ACCELERATE)
        else()
            message(WARNING "Accelerate framework not found")
        endif()
    endif()
endif()

target_compile_definitions(whisper PUBLIC DART_SHARED_LIB)
target_compile_definitions(ggml PUBLIC ${WHISPER_EXTRA_FLAGS})

target_link_libraries(whisper PRIVATE whisper_lib ggml ${WHISPER_EXTRA_LIBS} ${CMAKE_THREAD_LIBS_INIT})
