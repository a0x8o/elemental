#
#  Copyright 2009-2016, Jack Poulson
#  All rights reserved.
#
#  This file is part of Elemental and is under the BSD 2-Clause License,
#  which can be found in the LICENSE file in the root directory, or at
#  http://opensource.org/licenses/BSD-2-Clause
#
include(CheckCXXSourceCompiles)

find_package(MPI 3.0 REQUIRED COMPONENTS CXX)
if (MPI_CXX_FOUND)
  if (NOT TARGET MPI::MPI_CXX)
    add_library(MPI::MPI_CXX INTERFACE IMPORTED)
    if (MPI_CXX_COMPILE_FLAGS)
      separate_arguments(_MPI_CXX_COMPILE_OPTIONS UNIX_COMMAND
        "${MPI_CXX_COMPILE_FLAGS}")
      set_property(TARGET MPI::MPI_CXX PROPERTY
        INTERFACE_COMPILE_OPTIONS "${_MPI_CXX_COMPILE_OPTIONS}")
    endif()

    if(MPI_CXX_LINK_FLAGS)
      separate_arguments(_MPI_CXX_LINK_LINE UNIX_COMMAND
        "${MPI_CXX_LINK_FLAGS}")
    endif()

    set_property(TARGET MPI::MPI_CXX APPEND PROPERTY
      INTERFACE_LINK_LIBRARIES "${MPI_CXX_LIBRARIES}")

    set_property(TARGET MPI::MPI_CXX APPEND PROPERTY
      LINK_FLAGS "${_MPI_CXX_LINK_LINE}")

    set_property(TARGET MPI::MPI_CXX APPEND PROPERTY
      INTERFACE_INCLUDE_DIRECTORIES "${MPI_CXX_INCLUDE_PATH}")

  endif (NOT TARGET MPI::MPI_CXX)
else()
  message(FATAL_ERROR "MPI CXX compiler was not found and is required")
endif()

# Fix the imported target

# FIXME (trb): We should split the library into language-specific
# targets. That is, the .cu files should never need MPI linkage or
# OpenMP, so they should be built into a separate target without
# MPI::MPI_CXX or OpenMP::OpenMP_CXX "linkage".
get_target_property(
  __mpi_compile_options MPI::MPI_CXX INTERFACE_COMPILE_OPTIONS)
if (__mpi_compile_options)
  set_property(TARGET MPI::MPI_CXX PROPERTY
    INTERFACE_COMPILE_OPTIONS
    $<$<COMPILE_LANGUAGE:CXX>:${__mpi_compile_options}>)
  unset(__mpi_compile_options)
endif ()

get_property(_TMP_MPI_LINK_LIBRARIES TARGET MPI::MPI_CXX
  PROPERTY INTERFACE_LINK_LIBRARIES)
foreach(lib IN LISTS _TMP_MPI_LINK_LIBRARIES)
  if ("${lib}" MATCHES "-Wl*")
    list(APPEND _MPI_LINK_FLAGS "${lib}")
  else()
    list(APPEND _MPI_LINK_LIBRARIES "${lib}")
  endif ()
endforeach()

#set_property(TARGET MPI::MPI_CXX PROPERTY LINK_FLAGS ${_MPI_LINK_FLAGS})
set_property(TARGET MPI::MPI_CXX
  PROPERTY INTERFACE_LINK_LIBRARIES ${_MPI_LINK_LIBRARIES})

set(CMAKE_REQUIRED_FLAGS "${MPI_CXX_COMPILE_FLAGS}")
set(CMAKE_REQUIRED_LINKER_FLAGS "${MPI_CXX_LINK_FLAGS} ${CMAKE_EXE_LINKER_FLAGS}")
set(CMAKE_REQUIRED_INCLUDES ${MPI_CXX_INCLUDE_PATH})
set(CMAKE_REQUIRED_LIBRARIES ${MPI_CXX_LIBRARIES})

# These are a few checks to determine if we're using one of the common
# MPI libraries. This might matter for some atrocities we might commit
# with respect to synchronizing CUDA streams with the MPI library.
set(MPI_IS_OPEN_MPI_VARIANT_CODE
  "#include <mpi.h>
   int main(int argc, char** argv)
   {
     int is_open_mpi = OPEN_MPI;
     return !is_open_mpi;
   }")
check_cxx_source_compiles("${MPI_IS_OPENMPI_VARIANT_CODE}"
  HYDROGEN_MPI_IS_OPENMPI)

set(MPI_IS_MVAPICH2_VARIANT_CODE
  "#include <mpi.h>
   #include <iostream>
   int main(int argc, char** argv)
   {
     std::cout << MVAPICH2_VERSION << std::endl;
     return 0;
   }")
check_cxx_source_compiles("${MPI_IS_MVAPICH2_VARIANT_CODE}"
  HYDROGEN_MPI_IS_MVAPICH2)

# Check for CUDA-aware MPI
set(HYDROGEN_ENSURE_HOST_MPI_BUFFERS ON)
if (HYDROGEN_HAVE_CUDA
    AND (HYDROGEN_MPI_IS_OPENMPI OR HYDROGEN_MPI_IS_MVAPICH2))

  if (HYDROGEN_MPI_IS_OPENMPI)
    set(MPI_IS_CUDA_AWARE_CODE
      "#include <mpi.h>
       #include <mpi-ext.h>
       int main(int argc, char** argv)
       {
         int has_mpi_support = MPIX_CUDA_AWARE_SUPPORT;
         return !has_mpi_support;
       }")
    check_cxx_source_compiles("${MPI_IS_CUDA_AWARE_CODE}"
      HYDROGEN_HAVE_CUDA_AWARE_MPI)

  elseif (HYDROGEN_MPI_IS_MVAPICH2)
    set(MPI_IS_CUDA_AWARE_CODE
      "#include <mpi.h>
       #include <cuda_runtime.h>
       extern cudaStream_t stream_d2h;
       int main(int argc, char** argv)
       {
         if (stream_d2h)
           return 0;
         else
           return 0;
       }")
    set(CMAKE_REQUIRED_INCLUDES ${MPI_CXX_INCLUDE_PATH} ${CUDA_INCLUDE_DIRS})
    check_cxx_source_compiles("${MPI_IS_CUDA_AWARE_CODE}"
      HYDROGEN_HAVE_CUDA_AWARE_MPI)

  endif ()

  if (NOT HYDROGEN_HAVE_CUDA_AWARE_MPI)
    message(STATUS
      "Cannot detect CUDA-aware MPI. Support disabled.")
  else()
    if (NOT Hydrogen_AVOID_CUDA_AWARE_MPI)
      message(STATUS "Assuming CUDA-aware MPI.")
      set(HYDROGEN_ENSURE_HOST_MPI_BUFFERS OFF)
    else ()
      message(STATUS "CUDA-aware MPI detected but not used.")
    endif ()
  endif ()
endif ()

set(CMAKE_REQUIRED_FLAGS)
set(CMAKE_REQUIRED_LINKER_FLAGS)
set(CMAKE_REQUIRED_INCLUDES)
set(CMAKE_REQUIRED_LIBRARIES)
