# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  Haar.cpp
  # HatanoNelson.cpp
  HermitianUniformSpectrum.cpp
  # NormalUniformSpectrum.cpp
  # UniformHelmholtzGreens.cpp
  # Wigner.cpp
  )

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
