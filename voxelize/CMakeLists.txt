cmake_minimum_required(VERSION 3.10.0)
project(dense)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)

# Setup Python and Pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# If pybind11 is not found, try to find it via python
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE _pybind11_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

if(_pybind11_dir)
    list(APPEND CMAKE_PREFIX_PATH "${_pybind11_dir}")
endif()

find_package(pybind11 REQUIRED)

# Setup Eigen3
find_package(Eigen3 QUIET)
if(TARGET Eigen3::Eigen)
    message(STATUS "Found Eigen3 via find_package")
else()
    message(STATUS "Eigen3 target not found, trying standard include paths")
    if(EXISTS "/usr/include/eigen3")
        include_directories("/usr/include/eigen3")
    else()
        message(WARNING "Eigen3 headers not found in /usr/include/eigen3. Please install eigen3 or set EIGEN3_INCLUDE_DIR")
    endif()
endif()

# Output settings
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

pybind11_add_module(dense dense.cpp)

if(TARGET Eigen3::Eigen)
    target_link_libraries(dense PRIVATE Eigen3::Eigen)
endif()
