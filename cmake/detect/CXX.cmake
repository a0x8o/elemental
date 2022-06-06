#
#  Copyright 2009-2016, Jack Poulson
#  All rights reserved.
#
#  This file is part of Elemental and is under the BSD 2-Clause License,
#  which can be found in the LICENSE file in the root directory, or at
#  http://opensource.org/licenses/BSD-2-Clause
#
include(CheckCXXSourceCompiles)

# "restrict" support
set(RESTRICT_CODE "int main() { int* RESTRICT a; return 0; }")
set(CMAKE_REQUIRED_DEFINITIONS "-DRESTRICT=__restrict__")
check_cxx_source_compiles("${RESTRICT_CODE}" EL_HAVE___restrict__)
set(CMAKE_REQUIRED_DEFINITIONS "-DRESTRICT=__restrict")
check_cxx_source_compiles("${RESTRICT_CODE}" EL_HAVE___restrict)
set(CMAKE_REQUIRED_DEFINITIONS "-DRESTRICT=restrict")
check_cxx_source_compiles("${RESTRICT_CODE}" EL_HAVE_restrict)
if(EL_HAVE___restrict__)
  set(EL_RESTRICT "__restrict__")
  message(STATUS "Using __restrict__ keyword.")
elseif(EL_HAVE___restrict)
  set(EL_RESTRICT "__restrict")
  message(STATUS "Using __restrict keyword.")
elseif(EL_HAVE_restrict)
  set(EL_RESTRICT "restrict")
  message(STATUS "Using restrict keyword.")
else()
  set(EL_RESTRICT "")
  message(STATUS "Could not find a restrict keyword.")
endif()

# __PRETTY_FUNCTION__ support
set(PRETTY_FUNCTION_CODE
    "#include <iostream>
     int main()
     {
         std::cout << __PRETTY_FUNCTION__ << std::endl;
         return 0;
     }")
check_cxx_source_compiles("${PRETTY_FUNCTION_CODE}" EL_HAVE_PRETTY_FUNCTION)

unset(CMAKE_REQUIRED_FLAGS)
unset(CMAKE_REQUIRED_DEFINITIONS)
