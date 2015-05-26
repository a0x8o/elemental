#
#  Copyright 2009-2015, Jack Poulson
#  All rights reserved.
#
#  This file is part of Elemental and is under the BSD 2-Clause License,
#  which can be found in the LICENSE file in the root directory, or at
#  http://opensource.org/licenses/BSD-2-Clause
#
include(ElCheckFunctionExists)
include(CheckCXXSourceCompiles)

# Default locations (currently, Linux-centric) for searching for math libs
# ========================================================================
# NOTE: The variables BLAS_ROOT, LAPACK_ROOT, MATH_ROOT, BLAS_PATH, LAPACK_PATH,
#       and MATH_PATH are all optional and to ease the process
set(MATH_PATHS /usr/lib
               /usr/lib64
               /usr/local/lib
               /usr/local/lib64
               /usr/lib/openmpi/lib
               /usr/lib/gcc/x86_64-linux-gnu/4.8
               /usr/lib/gcc/x86_64-linux-gnu/4.9
               /lib/x86_64-linux-gnu
               /usr/lib/x86_64-linux-gnu
               /usr/lib/openblas-base
               /usr/lib64/openblas-base
               ${BLAS_ROOT}/lib
               ${BLAS_ROOT}/lib64
               ${LAPACK_ROOT}/lib
               ${LAPACK_ROOT}/lib64
               ${MATH_ROOT}/lib
               ${MATH_ROOT}/lib64
               ${BLAS_PATH}
               ${LAPACK_PATH}
               ${MATH_PATH})

# Check for BLAS and LAPACK support
# =================================
if(EL_HYBRID)
  set(MATH_DESC "Threaded BLAS/LAPACK link flags")
else()
  set(MATH_DESC "Unthreaded BLAS/LAPACK link flags")
endif()
if(MATH_LIBS)
  message(STATUS "Will attempt to use user-defined MATH_LIBS=${MATH_LIBS}")
elseif(NOT EL_DISABLE_SCALAPACK)
  include(external_projects/ElMath/ScaLAPACK)
  if(EL_HAVE_SCALAPACK)
    set(MATH_LIBS ${SCALAPACK_LIBS})  
  endif()
endif()

if(APPLE AND NOT MATH_LIBS)
  # Attempt to find/build BLIS+LAPACK if prompted
  # ---------------------------------------------
  if(EL_PREFER_BLIS_LAPACK)
    include(external_projects/ElMath/BLIS_LAPACK)
    if(EL_HAVE_BLIS_LAPACK)
      set(MATH_LIBS ${BLIS_LAPACK_LIBS})
      message("Will use BLIS+LAPACK via MATH_LIBS=${MATH_LIBS}")
    endif()
  endif()

  # Attempt to find/build OpenBLAS if prompted
  # ------------------------------------------
  if(NOT MATH_LIBS AND EL_PREFER_OPENBLAS)
    include(external_projects/ElMath/OpenBLAS)
    if(EL_HAVE_OPENBLAS)
      set(MATH_LIBS ${OPENBLAS_LIBS})
      message("Will use OpenBLAS+LAPACK via MATH_LIBS=${MATH_LIBS}")
    endif()
  endif()

  # Default to vecLib (older) or Accelerate (newer)
  # -----------------------------------------------
  if(NOT MATH_LIBS)
    set(CMAKE_REQUIRED_LIBRARIES "-framework vecLib")
    El_check_function_exists(dpotrf  EL_HAVE_DPOTRF_VECLIB)
    El_check_function_exists(dpotrf_ EL_HAVE_DPOTRF_POST_VECLIB)
    set(CMAKE_REQUIRED_LIBRARIES "-framework Accelerate")
    El_check_function_exists(dpotrf  EL_HAVE_DPOTRF_ACCELERATE)
    El_check_function_exists(dpotrf_ EL_HAVE_DPOTRF_POST_ACCELERATE)
    set(CMAKE_REQUIRED_LIBRARIES)
    if(EL_HAVE_DPOTRF_VECLIB OR EL_HAVE_DPOTRF_POST_VECLIB)
      set(MATH_LIBS "-framework vecLib" CACHE STRING ${MATH_DESC})
      message(STATUS "Using Apple vecLib framework.")
    elseif(EL_HAVE_DPOTRF_ACCELERATE OR EL_HAVE_DPOTRF_POST_ACCELERATE)
      set(MATH_LIBS "-framework Accelerate" CACHE STRING ${MATH_DESC})
      message(STATUS "Using Apple Accelerate framework.")
    endif()
  endif()
endif()

if(NOT MATH_LIBS)
  # Attempt to find/build OpenBLAS unless requested not to
  # ------------------------------------------------------
  if(NOT EL_DISABLE_OPENBLAS)
    include(external_projects/ElMath/OpenBLAS)
    if(EL_HAVE_OPENBLAS)
      set(MATH_LIBS ${OPENBLAS_LIBS})
      message("Will use OpenBLAS via MATH_LIBS=${MATH_LIBS}")
    endif()
  endif()

  # Attempt to find/build BLIS+LAPACK unless requested not to
  # ---------------------------------------------------------
  if(NOT MATH_LIBS AND NOT EL_DISABLE_BLIS_LAPACK)
    include(external_projects/ElMath/BLIS_LAPACK)
    if(EL_HAVE_BLIS_LAPACK)
      set(MATH_LIBS ${BLIS_LAPACK_LIBS})
      message("Will use BLIS via MATH_LIBS=${MATH_LIBS}")
    endif()
  endif()

  # Look for reference implementations of BLAS+LAPACK
  # -------------------------------------------------
  if(NOT MATH_LIBS)
    set(REFERENCE_REQUIRED LAPACK BLAS)
    find_library(BLAS_LIB NAMES blas PATHS ${MATH_PATHS})
    find_library(LAPACK_LIB NAMES lapack reflapack PATHS ${MATH_PATHS})
    set(REFERENCE_FOUND TRUE)
    foreach(NAME ${REFERENCE_REQUIRED})
      if(${NAME}_LIB)
        message(STATUS "Found ${NAME}_LIB: ${${NAME}_LIB}")
        list(APPEND MATH_LIBS ${${NAME}_LIB})
      else()
        message(STATUS "Could not find ${NAME}_LIB")
        set(MATH_LIBS "")
        set(REFERENCE_FOUND FALSE)
      endif()
    endforeach()
    if(REFERENCE_FOUND)
      message(WARNING "Using reference BLAS/LAPACK; performance will be poor")
    else()
      message(FATAL_ERROR "Could not find BLAS/LAPACK. Please specify MATH_LIBS")
    endif()
  endif()
endif()

# Check/predict the BLAS and LAPACK underscore conventions
# ========================================================
if(NOT EL_BUILT_BLIS_LAPACK AND NOT EL_BUILT_OPENBLAS)
  set(CMAKE_REQUIRED_FLAGS "${MPI_C_COMPILE_FLAGS}")
  set(CMAKE_REQUIRED_LINKER_FLAGS "${MPI_LINK_FLAGS} ${CMAKE_EXE_LINKER_FLAGS}")
  set(CMAKE_REQUIRED_INCLUDES ${MPI_C_INCLUDE_PATH})
  set(CMAKE_REQUIRED_LIBRARIES ${MATH_LIBS} ${MPI_C_LIBRARIES})
  # Check BLAS
  # ----------
  # NOTE: MATH_LIBS may involve MPI functionality (e.g., ScaLAPACK) and so
  #       MPI flags should be added for the detection
  if(EL_BLAS_SUFFIX)
    El_check_function_exists(daxpy${EL_BLAS_SUFFIX} EL_HAVE_DAXPY_SUFFIX)
    if(NOT EL_HAVE_DAXPY_SUFFIX)
      message(FATAL_ERROR "daxpy${EL_BLAS_SUFFIX} was not detected")
    endif()
    set(EL_HAVE_BLAS_SUFFIX TRUE)
  else()
    El_check_function_exists(daxpy  EL_HAVE_DAXPY)
    El_check_function_exists(daxpy_ EL_HAVE_DAXPY_POST)
    if(EL_HAVE_DAXPY)
      set(EL_HAVE_BLAS_SUFFIX FALSE)
    elseif(EL_HAVE_DAXPY_POST)
      set(EL_HAVE_BLAS_SUFFIX TRUE)
      set(EL_BLAS_SUFFIX _)
    else()
      message(FATAL_ERROR "Could not determine BLAS format.")
    endif()
  endif()
  # Check LAPACK
  # ------------
  if(EL_LAPACK_SUFFIX)
    El_check_function_exists(dpotrf${EL_BLAS_SUFFIX} EL_HAVE_DPOTRF_SUFFIX)
    if(NOT EL_HAVE_DPOTRF_SUFFIX)
      message(FATAL_ERROR "dpotrf${EL_LAPACK_SUFFIX} was not detected")
    endif()
    set(EL_HAVE_LAPACK_SUFFIX TRUE)
  else()
    El_check_function_exists(dpotrf  EL_HAVE_DPOTRF)
    El_check_function_exists(dpotrf_ EL_HAVE_DPOTRF_POST)
    if(EL_HAVE_DPOTRF)
      set(EL_HAVE_LAPACK_SUFFIX FALSE)
    elseif(EL_HAVE_DPOTRF_POST)
      set(EL_HAVE_LAPACK_SUFFIX TRUE)
      set(EL_LAPACK_SUFFIX _)
    else()
      message(FATAL_ERROR "Could not determine LAPACK format.")
    endif()
  endif()
  # Ensure that we have a relatively new version of LAPACK
  El_check_function_exists(dsyevr${EL_LAPACK_SUFFIX} EL_HAVE_DSYEVR)
  if(NOT EL_HAVE_DSYEVR)
    message(FATAL_ERROR "LAPACK is missing dsyevr${EL_LAPACK_SUFFIX}")
  endif()

  # Check for libFLAME support
  # ==========================
  El_check_function_exists(FLA_Bsvd_v_opd_var1 EL_HAVE_FLA_BSVD)

  # Check for ScaLAPACK support
  # ===========================
  if(NOT EL_DISABLE_SCALAPACK)
    if(EL_SCALAPACK_SUFFIX)
      El_check_function_exists(pdsyngst${EL_SCALAPACK_SUFFIX} EL_HAVE_PDSYNGST)
      El_check_function_exists(Csys2blacs_handle EL_HAVE_CSYS2BLACS)
      if(NOT EL_HAVE_PDSYNGST OR NOT EL_HAVE_CSYS2BLACS)
        message(WARNING "Could not find pdsyngst${EL_SCALAPACK_SUFFIX} or Csys2blacs_handle")
        set(EL_HAVE_SCALAPACK FALSE)
      else()
        set(EL_HAVE_SCALAPACK TRUE)
        set(EL_HAVE_SCALAPACK_SUFFIX TRUE)
      endif()
    else()
      # NOTE: pdsyngst was chosen because MKL ScaLAPACK only defines pdsyngst_,
      #       but not pdsyngst, despite defining both pdpotrf and pdpotrf_. 
      El_check_function_exists(pdsyngst  EL_HAVE_PDSYNGST)
      El_check_function_exists(pdsyngst_ EL_HAVE_PDSYNGST_POST)
      El_check_function_exists(Csys2blacs_handle EL_HAVE_CSYS2BLACS)
      if(EL_HAVE_PDSYNGST)
        El_check_function_exists(pdlaqr0 EL_HAVE_PDLAQR0)
        El_check_function_exists(pdlaqr1 EL_HAVE_PDLAQR1)
        if(NOT EL_HAVE_PDLAQR0 OR NOT EL_HAVE_PDLAQR1 OR 
           NOT EL_HAVE_CSYS2BLACS)
          message(STATUS "ScaLAPACK must support PDLAQR{0,1} and Csys2blacs_handle.")
        else()
          set(EL_HAVE_SCALAPACK TRUE)
          set(EL_HAVE_SCALAPACK_SUFFIX FALSE)
        endif()
      elseif(EL_HAVE_PDSYNGST_POST)
        El_check_function_exists(pdlaqr0_ EL_HAVE_PDLAQR0_POST)
        El_check_function_exists(pdlaqr1_ EL_HAVE_PDLAQR1_POST)
        if(NOT EL_HAVE_PDLAQR0_POST OR NOT EL_HAVE_PDLAQR1_POST OR
           NOT EL_HAVE_CSYS2BLACS)
          message(STATUS "ScaLAPACK must support PDLAQR{0,1} and Csys2blacs_handle.")
        else()
          set(EL_HAVE_SCALAPACK TRUE)
          set(EL_HAVE_SCALAPACK_SUFFIX TRUE)
          set(EL_SCALAPACK_SUFFIX _)
        endif()
      else()
        message(STATUS "Did not detect ScaLAPACK")
      endif()
    endif()
  endif()

  # Clean up the requirements since they cause problems in other Find packages,
  # such as FindThreads
  set(CMAKE_REQUIRED_FLAGS)
  set(CMAKE_REQUIRED_LINKER_FLAGS)
  set(CMAKE_REQUIRED_INCLUDES)
  set(CMAKE_REQUIRED_LIBRARIES)
endif()

# Check for quad-precision support
# ================================
if(NOT EL_DISABLE_QUAD)
  find_library(QUADMATH_LIB NAMES quadmath PATHS ${MATH_PATHS})
  if(QUADMATH_LIB)
    set(CMAKE_REQUIRED_LIBRARIES ${QUADMATH_LIB})
    set(QUADMATH_CODE
      "#include <complex>
       #include <iostream>
       #include <quadmath.h>
       int main( int argc, char* argv[] )
       {
           __float128 a = 2.0q;

           char aStr[128];
           quadmath_snprintf( aStr, sizeof(aStr), \"%Q\", a );
           std::cout << aStr << std::endl;

           __complex128 y;
           std::complex<__float128> z;

           return 0;    
       }")
    check_cxx_source_compiles("${QUADMATH_CODE}" EL_HAVE_QUADMATH)
    if(EL_HAVE_QUADMATH)
      set(EL_HAVE_QUAD TRUE)
      list(APPEND MATH_LIBS ${QUADMATH_LIB})
      message(WARNING "The usage of libquadmath effectively moves the Elemental build from the permissive New BSD License to the GPL; if this is not acceptable, it is necessary to reconfigure with the 'EL_DISABLE_QUAD=ON' option")
    else()
      message(WARNING "Found libquadmath but could not use it in C++")
    endif()
    set(CMAKE_REQUIRED_LIBRARIES)
  endif()
endif()

if(EL_DISABLE_PARMETIS)
  include(external_projects/ElMath/METIS)
else()
  include(external_projects/ElMath/ParMETIS)
endif()