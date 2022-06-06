# Exports the following variables
#
#   VTUNE_INCLUDE_PATH
#   VTUNE_LIBRARY
#   VTUNE_LIBRARIES
#
# The following IMPORTED target is also created:
#
#   vtune::vtune
#
find_path(VTUNE_INCLUDE_PATH ittnotify.h
  HINTS ${VTUNE_DIR} $ENV{VTUNE_DIR}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
  DOC "The location of VTune headers.")
find_path(VTUNE_INCLUDE_PATH ittnotify.h)

find_library(VTUNE_LIBRARY NAMES libittnotify.a ittnotify
  HINTS ${VTUNE_DIR} $ENV{VTUNE_DIR}
  PATH_SUFFIXES lib64 lib
  NO_DEFAULT_PATH
  DOC "The location of VTune Static lib")
find_library(VTUNE_LIBRARY NAMES libittnotify.a ittnotify)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(VTUNE
  DEFAULT_MSG VTUNE_LIBRARY VTUNE_INCLUDE_PATH)

add_library(vtune::vtune INTERFACE IMPORTED)

if (VTUNE_FOUND)
  set_property(TARGET vtune::vtune PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES "${VTUNE_INCLUDE_PATH}")

  set_property(TARGET vtune::vtune PROPERTY
    INTERFACE_LINK_LIBRARIES "${VTUNE_LIBRARY}")

  set(VTUNE_LIBRARIES vtune::vtune)
endif (VTUNE_FOUND)
