cmake_minimum_required(VERSION 3.18)
project(seclora_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYBIND11_STATUS
)
if(NOT PYBIND11_STATUS EQUAL 0)
    message(FATAL_ERROR
        "pybind11 is unavailable for ${Python3_EXECUTABLE}; "
        "install it with: ${Python3_EXECUTABLE} -m pip install pybind11")
endif()
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 CONFIG REQUIRED)

set(SECLORA_MCL_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/mcl"
    CACHE PATH "Path to the pinned herumi/mcl checkout")
if(NOT EXISTS "${SECLORA_MCL_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "mcl checkout not found at ${SECLORA_MCL_DIR}; run build.sh first")
endif()
add_subdirectory("${SECLORA_MCL_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/mcl")

set(SECLORA_CORE_SOURCES
    core/ife_fh_core.cpp
    core/pc_mcfe_client.cpp
    core/pc_mcfe_server.cpp
)

pybind11_add_module(
    _seclora_native
    bindings.cpp
    session.cpp
    ${SECLORA_CORE_SOURCES}
)
target_include_directories(
    _seclora_native
    PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/core"
    "${SECLORA_MCL_DIR}/include"
)
target_compile_options(_seclora_native PRIVATE -O3)
target_link_libraries(_seclora_native PRIVATE mcl mclbn256 pthread)
set_target_properties(
    _seclora_native
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
