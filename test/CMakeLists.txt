# test/CMakeLists.txt

add_executable(crasm_tests)

target_sources(crasm_tests
    PRIVATE
        main_tests.cu     
)

set_target_properties(crasm_tests PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "80;86;89"
)

target_include_directories(crasm_tests 
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include 
        ${CMAKE_SOURCE_DIR}/src     
        ${CMAKE_SOURCE_DIR}/cuda
        ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(crasm_tests 
    PRIVATE
        crasmcore
        CUDA::cudart
        GTest::gtest
        GTest::gtest_main
)

target_compile_options(crasm_tests PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:
        -Wall 
        -Wextra 
        -Wno-unused-parameter
    >
    $<$<COMPILE_LANGUAGE:CUDA>:
        --extended-lambda
        -Wno-deprecated-gpu-targets
        -Xcompiler=-Wno-unused-variable
        -Xcudafe=--diag_suppress=550
    >
)

include(GoogleTest)
gtest_discover_tests(crasm_tests)