/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/

/*!
  @file Macros for instantiating element types.

  Before including this file, #define PROTO(T) to the instantiation code
  for element type T.  Also #define EL_ENABLE_* to enable instantiation
  for the matching element type.

  Usage example: see src/core/DistMatrix/ElementMatrix.cpp.

  Hydrogen need not support complex element types, so macros for
  instantiating them can probably be removed.
*/

#ifndef PROTO_INT
# define PROTO_INT(T) PROTO(T)
#endif

#ifndef PROTO_REAL
# define PROTO_REAL(T) PROTO(T)
#endif
#ifndef PROTO_FLOAT
# define PROTO_FLOAT PROTO_REAL(float)
#endif
#ifndef PROTO_DOUBLE
# define PROTO_DOUBLE PROTO_REAL(double)
#endif

#if defined(HYDROGEN_HAVE_QD) && defined(EL_ENABLE_DOUBLEDOUBLE)
#ifndef PROTO_DOUBLEDOUBLE
# define PROTO_DOUBLEDOUBLE PROTO_REAL(DoubleDouble)
#endif
#endif

#if defined(HYDROGEN_HAVE_QD) && defined(EL_ENABLE_QUADDOUBLE)
#ifndef PROTO_QUADDOUBLE
# define PROTO_QUADDOUBLE PROTO_REAL(QuadDouble)
#endif
#endif

#if defined(HYDROGEN_HAVE_QUADMATH) && defined(EL_ENABLE_QUAD)
#ifndef PROTO_QUAD
# define PROTO_QUAD PROTO_REAL(Quad)
#endif
#endif

#if defined(HYDROGEN_HAVE_MPC) && defined(EL_ENABLE_BIGINT)
#ifndef PROTO_BIGINT
# define PROTO_BIGINT PROTO_INT(BigInt)
#endif
#endif

#if defined(HYDROGEN_HAVE_MPC) && defined(EL_ENABLE_BIGFLOAT)
#ifndef PROTO_BIGFLOAT
# define PROTO_BIGFLOAT PROTO_REAL(BigFloat)
#endif
#endif

#if defined(HYDROGEN_HAVE_HALF) && defined(EL_ENABLE_HALF)
#ifndef PROTO_HALF
# define PROTO_HALF PROTO_REAL(cpu_half_type)
#endif
#endif

#ifndef PROTO_COMPLEX
# define PROTO_COMPLEX(T) PROTO(T)
#endif
#ifndef PROTO_COMPLEX_FLOAT
# define PROTO_COMPLEX_FLOAT PROTO_COMPLEX(Complex<float>)
#endif
#ifndef PROTO_COMPLEX_DOUBLE
# define PROTO_COMPLEX_DOUBLE PROTO_COMPLEX(Complex<double>)
#endif

#if defined(HYDROGEN_HAVE_QD) && defined(EL_ENABLE_DOUBLEDOUBLE)
#ifndef PROTO_COMPLEX_DOUBLEDOUBLE
# define PROTO_COMPLEX_DOUBLEDOUBLE PROTO_COMPLEX(Complex<DoubleDouble>)
#endif
#endif

#if defined(HYDROGEN_HAVE_QD) && defined(EL_ENABLE_QUADDOUBLE)
#ifndef PROTO_COMPLEX_QUADDOUBLE
# define PROTO_COMPLEX_QUADDOUBLE PROTO_COMPLEX(Complex<QuadDouble>)
#endif
#endif

#if defined(HYDROGEN_HAVE_QUADMATH) && defined(EL_ENABLE_QUAD)
#ifndef PROTO_COMPLEX_QUAD
# define PROTO_COMPLEX_QUAD PROTO_COMPLEX(Complex<Quad>)
#endif
#endif

#if defined(HYDROGEN_HAVE_MPC) && defined(EL_ENABLE_BIGFLOAT)
#ifndef PROTO_COMPLEX_BIGFLOAT
# define PROTO_COMPLEX_BIGFLOAT PROTO_COMPLEX(Complex<BigFloat>)
#endif
#endif

#if defined(HYDROGEN_HAVE_HALF) && defined(EL_ENABLE_HALF)
#ifndef PROTO_COMPLEX_HALF
# define PROTO_COMPLEX_HALF PROTO_COMPLEX(Complex<cpu_half_type>)
#endif
#endif

#ifndef EL_NO_INT_PROTO
PROTO_INT(Int)
#if defined(EL_ENABLE_BIGINT) && defined(HYDROGEN_HAVE_MPC)
PROTO_BIGINT
#endif
#endif

#ifndef EL_NO_REAL_PROTO
# if !defined(EL_NO_FLOAT_PROTO)
PROTO_FLOAT
# endif
# if !defined(EL_NO_DOUBLE_PROTO)
PROTO_DOUBLE
# endif
#if defined(EL_ENABLE_DOUBLEDOUBLE) && defined(HYDROGEN_HAVE_QD)
PROTO_DOUBLEDOUBLE
#endif
#if defined(EL_ENABLE_QUADDOUBLE) && defined(HYDROGEN_HAVE_QD)
PROTO_QUADDOUBLE
#endif
#if defined(EL_ENABLE_QUAD) && defined(HYDROGEN_HAVE_QUADMATH)
PROTO_QUAD
#endif
#if defined(EL_ENABLE_BIGFLOAT) && defined(HYDROGEN_HAVE_MPC)
PROTO_BIGFLOAT
#endif
#if defined(EL_ENABLE_HALF) && defined(HYDROGEN_HAVE_HALF)
PROTO_HALF
#endif
#endif

#if !defined(EL_NO_COMPLEX_PROTO)
# if !defined(EL_NO_COMPLEX_FLOAT_PROTO)
PROTO_COMPLEX_FLOAT
# endif
# if !defined(EL_NO_COMPLEX_DOUBLE_PROTO)
PROTO_COMPLEX_DOUBLE
# endif
#if defined(EL_ENABLE_DOUBLEDOUBLE) && defined(HYDROGEN_HAVE_QD)
PROTO_COMPLEX_DOUBLEDOUBLE
#endif
#if defined(EL_ENABLE_QUADDOUBLE) && defined(HYDROGEN_HAVE_QD)
PROTO_COMPLEX_QUADDOUBLE
#endif
#if defined(EL_ENABLE_QUAD) && defined(HYDROGEN_HAVE_QUADMATH)
PROTO_COMPLEX_QUAD
#endif
#if defined(EL_ENABLE_BIGFLOAT) && defined(HYDROGEN_HAVE_MPC)
PROTO_COMPLEX_BIGFLOAT
#endif
#if defined(EL_ENABLE_HALF) && defined(HYDROGEN_HAVE_HALF)
// For instantiating Complex<cpu_half_type>, which requires a lot of work.
// PROTO_COMPLEX_HALF
#endif
#endif

#undef PROTO
#undef PROTO_INT
#undef PROTO_BIGINT

#undef PROTO_REAL
#undef PROTO_FLOAT
#undef PROTO_DOUBLE
#undef PROTO_DOUBLEDOUBLE
#undef PROTO_QUADDOUBLE
#undef PROTO_QUAD
#undef PROTO_BIGFLOAT
#undef PROTO_HALF

#undef PROTO_COMPLEX
#undef PROTO_COMPLEX_FLOAT
#undef PROTO_COMPLEX_DOUBLE
#undef PROTO_COMPLEX_DOUBLEDOUBLE
#undef PROTO_COMPLEX_QUADDOUBLE
#undef PROTO_COMPLEX_QUAD
#undef PROTO_COMPLEX_BIGFLOAT

#undef EL_ENABLE_DOUBLEDOUBLE
#undef EL_ENABLE_QUADDOUBLE
#undef EL_ENABLE_QUAD
#undef EL_ENABLE_BIGINT
#undef EL_ENABLE_BIGFLOAT
#undef EL_ENABLE_HALF

#undef EL_NO_INT_PROTO
#undef EL_NO_REAL_PROTO
#undef EL_NO_FLOAT_PROTO
#undef EL_NO_DOUBLE_PROTO
#undef EL_NO_HALF_PROTO
#undef EL_NO_COMPLEX_PROTO
#undef EL_NO_COMPLEX_FLOAT_PROTO
#undef EL_NO_COMPLEX_DOUBLE_PROTO
#undef EL_NO_COMPLEX_DOUBLEDOUBLE_PROTO
#undef EL_NO_COMPLEX_QUADDOUBLE_PROTO
#undef EL_NO_COMPLEX_QUAD_PROTO
#undef EL_NO_COMPLEX_BIGFLOAT_PROTO
