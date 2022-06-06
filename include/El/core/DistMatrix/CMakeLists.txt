# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  AbstractDistMatrix.hpp
  Block.hpp
  ElementMatrix.hpp
  )

# Add the subdirectories
add_subdirectory(Block)
add_subdirectory(ElementMatrix)

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
