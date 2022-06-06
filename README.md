<<<<<<< HEAD
# Hydrogen
=======
<p align="left" style="padding: 20px">
<img src="https://github.com/elemental/elemental-web/raw/master/source/_static/elemental.png">
</p>
>>>>>>> 6eb15a0da (Update README.md)

Hydrogen is a fork of
[Elemental](https://github.com/elemental/elemental) used by
[LBANN](https://github.com/llnl/lbann). Hydrogen is a redux of the
Elemental functionality that has been ported to make use of GPGPU
accelerators. The supported functionality is essentially the core
infrastructure plus BLAS-1 and BLAS-3.

## Building

<<<<<<< HEAD
Hydrogen builds with a [CMake](https://cmake.org) (version 3.9.0 or
newer) build system. The build system respects the "normal" CMake
variables (`CMAKE_CXX_COMPILER`, `CMAKE_INSTALL_PREFIX`,
`CMAKE_BUILD_TYPE`, etc) in addition to the [Hydrogen-specific options
documented below](#hydrogen-cmake-options).
=======
### Deprecation notice
Elemental has not been maintained since 2016. But the project was [forked by Lawrence Livermore National Lab](https://github.com/LLNL/Elemental). The author stopped being interested in volunteering to develop MPI codes and no one has stepped up after three years.

**Software consists of teams of people. If you want people to continue developing a project after it ceases to be their personal interest, fund them for it.**

The developer is now volunteering time towards high-performance math software for workstations at [hodgestar.com](https://hodgestar.com).

### Documentation

The (now outdated) [documentation for Elemental](http://elemental.github.io/documentation) is built using [Sphinx](http://sphinx.pocoo.org) and the [Read the Docs Theme](http://docs.readthedocs.org/en/latest/theme.html)

### Unique features

Elemental supports a wide collection of sequential and distributed-memory
functionality, including sequential and distributed-memory support for the
datatypes:

- `float`, `El::Complex<float>`
- `double`, `El::Complex<double>`
- `El::DoubleDouble`, `El::Complex<El::DoubleDouble>` (on top of QD's *dd_real*)
- `El::QuadDouble`, `El::Complex<El::QuadDouble>` (on top of QD's *qd_real*)
- `El::Quad`, `El::Complex<El::Quad>` (on top of GCC's *__float128*)
- `El::BigFloat`, `El::Complex<El::BigFloat>` (on top of MPFR's *mpfr_t* and MPC's *mpc_t*)

**Linear algebra**:
* Dense and sparse-direct (generalized) Least Squares
  problems
    - Least Squares / Minimum Length
    - Tikhonov (and ridge) regression
    - Equality-constrained Least Squares
    - General (Gauss-Markov) Linear Models
* High-performance pseudospectral computation and visualization
* Aggressive Early Deflation Schur decompositions (currently sequential only)
* Blocked column-pivoted QR via Johnson-Lindenstrauss
* Quadratic-time low-rank Cholesky and LU modifications
* Bunch-Kaufman and Bunch-Parlett for accurate symmetric
  factorization
* LU and Cholesky with full pivoting
* Column-pivoted QR and interpolative/skeleton decompositions
* Quadratically Weighted Dynamic Halley iteration for the polar decomposition
* Many algorithms for Singular-Value soft-Thresholding (SVT)
* Tall-skinny QR decompositions
* Hermitian matrix functions
* Prototype Spectral Divide and Conquer Schur decomposition and Hermitian EVD
* Sign-based Lyapunov/Ricatti/Sylvester solvers
* Arbitrary-precision distributed SVD (QR and D&C support), (generalized) Hermitian EVPs (QR and D&C support), and Schur decompositions (e.g., via Aggressive Early Deflation)

**Convex optimization**:
* Dense and sparse Interior Point Methods for
  Linear, Quadratic, and Second-Order Cone Programs (**Note: Scalability for sparse IPMs will be lacking until more general sparse matrix distributions are introduced into Elemental**)
    - Basis Pursuit
    - Chebyshev Points
    - Dantzig selectors
    - LASSO / Basis Pursuit Denoising
    - Least Absolute Value regression
    - Non-negative Least Squares
    - Support Vector Machines
    - (1D) Total Variation
* Jordan algebras over products of Second-Order Cones
* Various prototype dense Alternating Direction Method of Multipliers routines
    - Sparse inverse covariance selection
    - Robust Principal Component Analysis
* Prototype alternating direction Non-negative Matrix Factorization

**Lattice reduction**:
* An extension of [Householder-based LLL](http://perso.ens-lyon.fr/damien.stehle/HLLL.html) to real and complex linearly-dependent bases (currently sequential only)
* Generalizations of [BKZ 2.0](http://link.springer.com/chapter/10.1007%2F978-3-642-25385-0_1) to complex bases (currently sequential only)
 incorporating ["y-sparse" enumeration](https://eprint.iacr.org/2014/980)
* Integer images/kernels and relation-finding (currently sequential only)

### The current development roadmap

**Core data structures**:
* (1a) Eliminate `DistMultiVec` in favor of the newly extended `DistMatrix`
* (1b) Extend `DistSparseMatrix` to support elementwise and blockwise 2D distributions

**Linear algebra**:
* (2a) Distributed iterative refinement tailored to two right-hand sides \[weakly depends on (1a)\]
* (2b) Extend black-box iterative refinement to `DistMatrix`
* (2c) Incorporate iterative refinement into linear solvers via optional control
  structure \[weakly depends upon (2b)\]
* (2d) Support for the Fix-Heiberger method for accurate generalized Hermitian-definite EVPs

**Convex optimization**:
* (3a) Add support for homogeneous self-dual embeddings \[weakly depends on (2a)\]
* (3b) Enhance sparse scalability via low edge-degree plus low-rank 
  decompositions \[depends on (1b); weakly depends on (1a)\]
* (3c) Distributed sparse semidefinite programs via chordal decompositions \[weakly depends on (3b)\]

### License

The vast majority of Elemental is distributed under the terms of the
[New BSD License](http://www.opensource.org/licenses/bsd-license.php).
Please see the [debian/copyright](https://github.com/elemental/Elemental/blob/master/debian/copyright) file for an overview of the copyrights and licenses for
the files in the library.

The optional external dependency
[METIS](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview)
is distributed under the (equally permissive)
[Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0.html),
though
[ParMETIS](http://glaros.dtc.umn.edu/gkhome/metis/parmetis/overview)
can only be used for research purposes (and can be easily disabled).
[libquadmath](https://gcc.gnu.org/onlinedocs/libquadmath/) is 
distributed under the terms of the [GNU Lesser General Public License, version 2.1 or later](http://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html),
while,
[QD](http://crd-legacy.lbl.gov/~dhbailey/mpdist/) is distributed under the
terms of the [LBNL-BSD-License](http://crd.lbl.gov/~dhbailey/mpdist/LBNL-BSD-License.doc).
>>>>>>> 6eb15a0da (Update README.md)

### Dependencies

The most basic build of Hydrogen requires only:

+ [CMake](https://cmake.org): Version 3.9.0 or newer.

+ A C++11-compliant compiler.

+ MPI 3.0-compliant MPI library.

+ [BLAS](http://www.netlib.org/blas/): Provides basic linear
  algebra kernels for the CPU code path.

+ [LAPACK](http://www.netlib.org/lapack/): Provides a few utility
  functions (norms and 2D copies, e.g.). This could be demoted to
  "optional" status with little effort.
  
Optional dependencies of Hydrogen include:

+ [Aluminum](https://github.com/llnl/aluminum): Provides asynchronous
  blocking and non-blocking communication routines with an MPI-like
  syntax. The use of Aluminum is **highly** recommended.

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

+ `Hydrogen_ENABLE_ALUMINUM` (Default: `OFF`): Enable the
  [Aluminum](https://github.com/llnl/aluminum) library for
  asynchronous device-aware communication. The use of this library is
  **highly** recommended for CUDA-enabled builds.

+ `Hydrogen_ENABLE_CUDA` (Default: `OFF`): Enable CUDA support in the
  library. This enables the device type `El::Device::GPU` and allows
  memory to reside on CUDA-aware GPGPUs.

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
        -DHydrogen_ENABLE_ALUMINUM=ON \
        -DCUB_DIR=/path/to/cub \
        -DAluminum_DIR=/path/to/aluminum \
        /path/to/hydrogen
    ninja install
```

## Reporting issues

Issues should be reported [on
Github](https://github.com/llnl/elemental/issues/new).
