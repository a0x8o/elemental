# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  AbstractDistMatrix.cpp
  Block.cpp
  ElementMatrix.cpp
  )

# Add the subdirectories
add_subdirectory(Block)
add_subdirectory(ElementMatrix)

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
