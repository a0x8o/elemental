# Sets the following variables
#
#   NVTX_FOUND
#   NVTX_LIBRARY
#   NVTX_LIBRARIES
#
# Defines the following imported target:
#
#   cuda::nvtx
#

find_library(NVTX_LIBRARY nvToolsExt
  HINTS ${NVTX_DIR} $ENV{NVTX_DIR} ${CUDA_TOOLKIT_ROOT_DIR} ${CUDA_SDK_ROOT_DIR}
  PATH_SUFFIXES lib64 lib
  DOC "The nvtx library."
  NO_DEFAULT_PATH)
find_library(NVTX_LIBRARY nvToolsExt)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NVTX
  DEFAULT_MSG NVTX_LIBRARY)

add_library(cuda::nvtx INTERFACE IMPORTED)

if (NVTX_FOUND)
  set_property(TARGET cuda::nvtx PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES "${CUDA_INCLUDE_DIRS}")

  set_property(TARGET cuda::nvtx PROPERTY
    INTERFACE_LINK_LIBRARIES "${NVTX_LIBRARY}")

  set(NVTX_LIBRARIES cuda::nvtx)
endif (NVTX_FOUND)
