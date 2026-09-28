# Stage-3 revival tests: C++ and Fortran APIs vs pylibnxc reference cases.
#   -DNXC_TEST_PYTHON=/path/to/python   (needs torch; pyscf optional for molecular cases)
#   -DNXC_TEST_EXTRA_MODELS=/path/to/models   (optional: exported models, <dir>/<NAME>/xc)
find_package(Python3 COMPONENTS Interpreter)
set(NXC_TEST_PYTHON "${Python3_EXECUTABLE}" CACHE FILEPATH "Python with torch (and pyscf) for reference data")
set(NXC_TEST_EXTRA_MODELS "" CACHE PATH "Directory with additional (exported) libnxc models")

add_executable(test_cases test_cases.cpp)
target_link_libraries(test_cases PRIVATE nxc)
add_executable(bench bench.cpp)          # throughput, not a test
target_link_libraries(bench PRIVATE nxc)

set(_refdir ${CMAKE_CURRENT_BINARY_DIR}/refdata)
add_test(NAME revival_refdata
         COMMAND ${NXC_TEST_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/gen_reference.py ${_refdir} ${NXC_TEST_EXTRA_MODELS})
set_tests_properties(revival_refdata PROPERTIES FIXTURES_SETUP refdata ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")

add_test(NAME revival_cxx
         COMMAND ${NXC_TEST_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/run_cases.py ${_refdir}/cases.txt $<TARGET_FILE:test_cases> cxx)
set_tests_properties(revival_cxx PROPERTIES FIXTURES_REQUIRED refdata ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")

if(NXC_FORTRAN)
  add_executable(test_fortran test_fortran.f90)
  target_link_libraries(test_fortran PRIVATE nxc_fortran nxc)
  add_test(NAME revival_fortran
           COMMAND ${NXC_TEST_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/run_cases.py ${_refdir}/cases.txt $<TARGET_FILE:test_fortran> fortran)
  set_tests_properties(revival_fortran PROPERTIES FIXTURES_REQUIRED refdata ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
endif()
