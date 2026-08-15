# src/CMakeLists.txt

target_sources(crasmcore 
    PRIVATE
        crasm.cpp
        # main.cpp
)


add_subdirectory(assembler)
add_subdirectory(lexer)
add_subdirectory(transfer)