# /CMakeLists.txt
cmake_minimum_required(VERSION 3.30)

# set(CMAKE_CUDA_HOST_COMPILER clang++-21)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

project(crasm VERSION 0.1.0 LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(crasmcore STATIC)

target_include_directories(crasmcore 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/cuda
)

set_target_properties(crasmcore PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "80;86;89"
)

target_compile_options(crasmcore PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:
        -Wall 
        -Wextra 
        -Wno-unused-parameter
    >

    $<$<COMPILE_LANGUAGE:CUDA>:
        --extended-lambda
        -Wno-deprecated-gpu-targets
        -Xcompiler=-Wno-unused-variable
        # -Xcudafe=--diag_suppress=550
    >
)


target_compile_definitions(crasmcore 
    PRIVATE
        $<$<CONFIG:Debug>:CRASM_DEBUG>
)

add_subdirectory(src)

target_link_libraries(crasmcore 
    PUBLIC
        # ${CUDAToolkit_LIBRARIES}
        CUDA::cudart
)


add_executable(crasm src/main.cpp)
target_link_libraries(crasm PRIVATE crasmcore)
set_target_properties(crasm PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

option(BUILD_TESTING "Build tests" OFF)

if(BUILD_TESTING)
    enable_testing()
    find_package(GTest REQUIRED)
    add_subdirectory(test)
endif()


# option(BUILD_TESTING "Build tests" OFF)

# if(BUILD_TESTING)
#     enable_testing()
#     find_package(GTest REQUIRED)

#     add_executable(tests_crasm test/main.cu)
#     set(TEST_EXEC_NAME "tests_crasm")

#     add_subdirectory(test)

#     set_target_properties(${TEST_EXEC_NAME} PROPERTIES
#         CUDA_STANDARD 20
#         CUDA_STANDARD_REQUIRED ON
#         CUDA_SEPARABLE_COMPILATION ON
#         CUDA_ARCHITECTURES "80;86;89"
#         )

#     target_include_directories(${TEST_EXEC_NAME} PRIVATE
#         ${CUDAToolkit_INCLUDE_DIRS}
#         ${CMAKE_SOURCE_DIR}/include
#         )

#     target_compile_options(${TEST_EXEC_NAME} PRIVATE 
#         -Wall 
#         -Wextra
#         -Wno-unused-parameter
#         -Wno-deprecated-gpu-targets # for 6.0 compute p100
#         -Xcompiler -Wno-unused-variable
#         -Xcudafe --diag_suppress=550
#         )

#     target_link_libraries(${TEST_EXEC_NAME} PRIVATE 
#         crasm_lib
#         ${CUDAToolkit_LIBRARIES}
#         GTest::gtest
#         GTest::gtest_main
#         )

#     add_test(NAME crasm_tests COMMAND ${TEST_EXEC_NAME})

# endif()