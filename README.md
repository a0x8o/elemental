# Hydrogen

Hydrogen is a fork of
[Elemental](https://github.com/elemental/elemental) used by
[LBANN](https://github.com/llnl/lbann). Hydrogen is a redux of the
Elemental functionality that has been ported to make use of GPGPU
accelerators. The supported functionality is essentially the core
infrastructure plus BLAS-1 and BLAS-3.

## Building

Hydrogen builds with a [CMake](https://cmake.org) (version 3.21 or
newer) build system. The build system respects the "normal" CMake
variables (`CMAKE_CXX_COMPILER`, `CMAKE_INSTALL_PREFIX`,
`CMAKE_BUILD_TYPE`, etc) in addition to the [Hydrogen-specific options
documented below](#hydrogen-cmake-options).

### Dependencies

The most basic build of Hydrogen requires only:

+ [CMake](https://cmake.org): Version 3.21 or newer.

+ A C++17-compliant compiler.

+ MPI 3.0-compliant MPI library.

+ [Aluminum](https://github.com/llnl/aluminum): Version 1.0.0 or
  newer (1.3.0 or newer for in-place `SendRecv`). Provides
  asynchronous blocking and non-blocking communication routines with
  an MPI-like syntax. Configuration fails if Aluminum is not found;
  point CMake at it with `Aluminum_ROOT` or `CMAKE_PREFIX_PATH`. A
  CPU-only Aluminum (`-DALUMINUM_ENABLE_CUDA=OFF
  -DALUMINUM_ENABLE_NCCL=OFF`) is sufficient for CPU-only builds of
  Hydrogen.

+ [BLAS](http://www.netlib.org/blas/): Provides basic linear
  algebra kernels for the CPU code path.

+ [LAPACK](http://www.netlib.org/lapack/): Provides a few utility
  functions (norms and 2D copies, e.g.). This could be demoted to
  "optional" status with little effort.
  
Optional dependencies of Hydrogen include:

+ [CUDA](https://developer.nvidia.com/cuda-zone): Version 9.2 or
  newer. Hydrogen primarily uses the runtime API and also grabs some
  features of NVML and NVPROF (if enabled).

+ [CUB](https://github.com/nvlabs/cub): Version 1.8.0 is
  recommended. This will become required for CUDA-enabled builds in
  the very near future.

+ [Half](https://half.sourceforge.net): Provides support for IEEE-754
  16-bit precision support. (*Note*: This is work in progress.)

+ [OpenMP](https://www.openmp.org): OpenMP 3.0 is probably sufficient
  for the limited use of the features in Hydrogen.

+ [ROCm/HIP](https://rocm.docs.amd.com): Enables the AMD GPU code
  path (with hipCUB and, optionally, roctracer).

+ [Catch2](https://github.com/catchorg/Catch2): Version 2.x. Required
  only for the unit tests (`Hydrogen_ENABLE_UNIT_TESTS`).

+ [VTune](https://software.intel.com/en-us/vtune): Proprietary
  profiler from Intel. May provide more detailed annotations to
  profiles of Hydrogen CPU code.

### Hydrogen CMake options

Some of the options are inherited from Elemental with `EL_` replaced
by `Hydrogen_`. Others are unique to Hydrogen. Supported options are:

+ `Hydrogen_AVOID_CUDA_AWARE_MPI` (Default: `OFF`): There is a very
  small amount of logic to try to detect CUDA-aware MPI (it should not
  give a false-positive but is likey to give a false negative). This
  option causes the library to ignore this and assume the MPI library
  is not CUDA-aware.

+ `Hydrogen_ENABLE_CUDA` (Default: `OFF`): Enable CUDA support in the
  library. This enables the device type `El::Device::GPU` and allows
  memory to reside on CUDA-aware GPGPUs.

+ `Hydrogen_ENABLE_ROCM` (Default: `OFF`): Enable ROCm/HIP support in
  the library. Mutually exclusive with `Hydrogen_ENABLE_CUDA`.

+ `Hydrogen_ENABLE_CUB` (Default: `Hydrogen_ENABLE_CUDA`): Only
  available if CUDA is enabled. This enables device memory management
  through a memory pool using [CUB](https://github.com/nvlabs/cub).

+ `Hydrogen_ENABLE_HALF` (Default: `OFF`): Enable IEEE-754 "binary16"
  16-bit precision floating point support through the [Half
  library](https://half.sourceforge.net).

+ `Hydrogen_ENABLE_BFLOAT16` (Default: `OFF`): This option is a
  placeholder. This will enable support for "bfloat16" 16-bit
  precision floating point arithmetic if/when that becomes a thing.

+ `Hydrogen_USE_64BIT_INTS` (Default: `OFF`): Use `long` as the
  default signed integer type within Hydrogen.

+ `Hydrogen_USE_64BIT_BLAS_INTS` (Default: `OFF`): Use `long` as the
  default signed integer type for interacting with BLAS libraries.

+ `Hydrogen_ENABLE_TESTING` (Default: `ON`): Build the test suite.

+ `Hydrogen_ENABLE_UNIT_TESTS` (Default: `OFF`): Build the
  Catch2-based unit tests.

+ `Hydrogen_ZERO_INIT` (Default: `OFF`): Initialize buffers to zero by
  default. There will obviously be a compute-time overhead.

+ `Hydrogen_ENABLE_NVPROF` (Default: `OFF`): Enable library
  annotations using the `nvtx` interface in CUDA.

+ `Hydrogen_ENABLE_VTUNE` (Default: `OFF`): Enable library annotations
  for use with Intel's VTune performance profiler.

+ `Hydrogen_ENABLE_SYNCHRONOUS_PROFILING` (Default: `OFF`):
  Synchronize computation at the beginning of profiling regions.

+ `Hydrogen_ENABLE_OPENMP` (Default: `OFF`): Enable OpenMP on-node
  parallelization primatives. OpenMP is used for CPU parallelization
  only; the device offload features of modern OpenMP are not used.

+ `Hydrogen_ENABLE_OMP_TASKLOOP` (Default: `OFF`): Use `omp taskloop`
  instead of `omp parallel for`. This is a highly experimental
  feature. Use with caution.

The following options are legacy options inherited from Elemental. The
related functionality is not tested regularly. The likely implication
of this statement is that nothing specific to this option has been
removed from what remains of Elemental but also that nothing specific
to these options has been added to any of the new features of
Hydrogen.

+ `Hydrogen_ENABLE_VALGRIND` (Default: `OFF`): Search for `valgrind`
  and enable related features if found.

+ `Hydrogen_ENABLE_QUADMATH` (Default: `OFF`): Search for the `quadmath`
  library and enable related features if found. This is for
  extended-precision computations.

+ `Hydrogen_ENABLE_QD` (Default: `OFF`): Search for the `QD` library
  and enable related features if found. This is for extended-precision
  computations.

+ `Hydrogen_ENABLE_MPC` (Default: `OFF`): Search for the GNU MPC
  library (requires MPFR and GMP as well) and enable related features
  if found. This is for extended precision.

+ `Hydrogen_USE_CUSTOM_ALLTOALLV` (Default: `OFF`): Avoid
  MPI_Alltoallv for performance reasons.

+ `Hydrogen_AVOID_COMPLEX_MPI` (Default: `OFF`): Avoid potentially
  buggy complex MPI routines.

+ `Hydrogen_USE_BYTE_ALLGATHERS` (Default: `OFF`): Avoid BG/P
  allgather performance bug.

+ `Hydrogen_CACHE_WARNINGS` (Default: `OFF`): Warns when using
  cache-unfriendly routines.

+ `Hydrogen_UNALIGNED_WARNINGS` (Default: `OFF`): Warn when performing
  unaligned redistributions.

+ `Hydrogen_VECTOR_WARNINGS` (Default: `OFF`): Warn when vector
  redistribution chances are missed.

### Example CMake invocation

The following builds a CUDA-enabled, CUB-enabled, Aluminum-enabled
version of Hydrogen:

```bash
    cmake -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_INSTALL_PREFIX=/path/to/my/install \
        -DHydrogen_ENABLE_CUDA=ON \
        -DHydrogen_ENABLE_CUB=ON \
        -DCUB_DIR=/path/to/cub \
        -DAluminum_DIR=/path/to/aluminum \
        /path/to/hydrogen
    ninja install
```

The following builds a CPU-only version of Hydrogen and runs the test
suite, as done in CI (`.github/workflows/ci.yml`):

```bash
    cmake -S /path/to/aluminum -B build-aluminum \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/path/to/aluminum-install \
        -DALUMINUM_ENABLE_CUDA=OFF \
        -DALUMINUM_ENABLE_NCCL=OFF
    cmake --build build-aluminum --parallel
    cmake --install build-aluminum

    cmake -S /path/to/hydrogen -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DAluminum_ROOT=/path/to/aluminum-install \
        -DHydrogen_ENABLE_TESTING=ON \
        -DHydrogen_ENABLE_UNIT_TESTS=ON
    cmake --build build --parallel
    ctest --test-dir build --output-on-failure
```

## Reporting issues

Issues should be reported [on
Github](https://github.com/a0x8o/elemental/issues/new).
