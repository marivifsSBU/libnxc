# CMake build for libnxc (C++ core, C and Fortran bindings).
#
#   cmake -S . -B build-cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
#   cmake --build build-cmake -j
#   ctest --test-dir build-cmake --output-on-failure
#
# libtorch can come from a pip/conda torch installation (as above) or from a libtorch
# zip (-DCMAKE_PREFIX_PATH=/path/to/libtorch).  Use the same torch version (or newer)
# as the one that exported the TorchScript models.
# The legacy Makefile/arch.make build in utils/ is kept for reference.

cmake_minimum_required(VERSION 3.18)
project(libnxc VERSION 0.3.0 LANGUAGES CXX)

option(NXC_BUILD_TESTS "Build the C++/Fortran tests" ON)
option(NXC_FORTRAN "Build the Fortran module and Fortran tests" ON)
option(NXC_MPI "Enable MPI support of the (NeuralXC) atomic functionals" OFF)

set(CMAKE_CXX_STANDARD 20)   # libtorch >= 2.10 requires C++20
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Torch REQUIRED)
get_filename_component(NXC_TORCH_LIBDIR "${TORCH_LIBRARY}" DIRECTORY)

if(NXC_FORTRAN)
  enable_language(Fortran)
endif()

add_library(nxc SHARED
  src/defaults.cpp
  src/gridfunc.cpp
  src/nxc_func.cpp
  src/atomicfunc.cpp
  src/nxc.cpp
  src/nxc_c.cpp
  src/nxc_f90.cpp)
target_include_directories(nxc PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:include/nxc>)
target_link_libraries(nxc PUBLIC ${TORCH_LIBRARIES})
separate_arguments(_nxc_torch_flags UNIX_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(nxc PUBLIC ${_nxc_torch_flags})
target_compile_definitions(nxc PRIVATE
  NXC_DEFAULTPATH_STR="${CMAKE_CURRENT_SOURCE_DIR}/models/")
if(NXC_MPI)
  target_compile_definitions(nxc PRIVATE MPI)
endif()
set_target_properties(nxc PROPERTIES
  BUILD_RPATH "${NXC_TORCH_LIBDIR}"
  INSTALL_RPATH "${NXC_TORCH_LIBDIR};${CMAKE_INSTALL_PREFIX}/lib")

if(NXC_FORTRAN)
  # ISO_C_BINDING interfaces to the Fortran API (module nxc)
  add_library(nxc_fortran STATIC src/nxc_mod.f90)
  set_target_properties(nxc_fortran PROPERTIES
    Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fortran_modules)
  target_include_directories(nxc_fortran PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/fortran_modules>)
  target_link_libraries(nxc_fortran PUBLIC nxc)
endif()

include(GNUInstallDirs)
install(TARGETS nxc LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES src/nxc.h src/nxc_c.h src/nxc_f90.h src/defaults.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/nxc)
if(NXC_FORTRAN)
  install(TARGETS nxc_fortran ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
  install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fortran_modules/
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/nxc)
endif()

if(NXC_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test/revival)
endif()
