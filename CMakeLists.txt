cmake_minimum_required(VERSION 3.15)
project(BlitzScale)

# Use C++17 (modern but widely supported)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable warnings
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -pedantic)
endif()

# Find threads (for std::thread)
find_package(Threads REQUIRED)

# Source files
set(SOURCES
    src/main.cpp
    src/network.cpp
    src/gpu.cpp
)

# Create executable
add_executable(blitzscale ${SOURCES})

# Add include directory
target_include_directories(blitzscale PRIVATE ${PROJECT_SOURCE_DIR}/include)

# Link threads
target_link_libraries(blitzscale Threads::Threads)
