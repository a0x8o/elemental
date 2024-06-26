/*
  Copyright (c) 2009-2016, Jack Poulson
  All rights reserved.

  This file is part of Elemental and is under the BSD 2-Clause License,
  which can be found in the LICENSE file in the root directory, or at
  http://opensource.org/licenses/BSD-2-Clause
*/
#include <El-lite.hpp>
#include <El/blas_like/level1.hpp>
#include <El/blas_like/level3.hpp>

#include "./Trrk/Local.hpp"
#include "./Trrk/NN.hpp"
#include "./Trrk/NT.hpp"
#include "./Trrk/TN.hpp"
#include "./Trrk/TT.hpp"
#include "core/Element/Complex/decl.hpp"

namespace El {

template<typename T>
void TrrkInternal(
    UpperOrLower uplo,
    Orientation orientA, Orientation orientB,
    T alpha, Matrix<T,Device::CPU> const& A, Matrix<T,Device::CPU> const& B,
    T beta, Matrix<T>& C)
{
    EL_DEBUG_CSE;
    ScaleTrapezoid(beta, uplo, C);
    if (orientA==NORMAL && orientB==NORMAL)
        trrk::TrrkNN(uplo, alpha, A, B, C);
    else if (orientA==NORMAL)
        trrk::TrrkNT(uplo, orientB, alpha, A, B, C);
    else if (orientB==NORMAL)
        trrk::TrrkTN(uplo, orientA, alpha, A, B, C);
    else
        trrk::TrrkTT(uplo, orientA, orientB, alpha, A, B, C);
}

#ifdef HYDROGEN_HAVE_MKL_GEMMT
template<typename T,typename=EnableIf<IsBlasScalar<T>>>
void TrrkMKL(
    UpperOrLower uplo,
    Orientation orientA, Orientation orientB,
    T alpha, Matrix<T,Device::CPU> const& A, Matrix<T,Device::CPU> const& B,
    T beta,        Matrix<T>& C)
{
    EL_DEBUG_CSE;
    const char uploChar = UpperOrLowerToChar(uplo);
    const char orientAChar = OrientationToChar(orientA);
    const char orientBChar = OrientationToChar(orientB);
    const auto n = C.Height();
    const auto k = orientA == NORMAL ? A.Width() : A.Height();
    mkl::Trrk
        (uploChar, orientAChar, orientBChar,
         n, k,
         alpha, A.LockedBuffer(), A.LDim(),
         B.LockedBuffer(), B.LDim(),
         beta,  C.Buffer(),       C.LDim());
}
#endif

template<typename T,typename=EnableIf<IsBlasScalar<T>>>
void TrrkHelper(
    UpperOrLower uplo,
    Orientation orientA, Orientation orientB,
    T alpha, Matrix<T,Device::CPU> const& A, Matrix<T,Device::CPU> const& B,
    T beta, Matrix<T>& C)
{
    EL_DEBUG_CSE;
#ifdef HYDROGEN_HAVE_MKL_GEMMT
    TrrkMKL(uplo, orientA, orientB, alpha, A, B, beta, C);
#else
    TrrkInternal(uplo, orientA, orientB, alpha, A, B, beta, C);
#endif
}

template<typename T,typename=DisableIf<IsBlasScalar<T>>,typename=void>
void TrrkHelper(
    UpperOrLower uplo,
    Orientation orientA, Orientation orientB,
    T alpha, Matrix<T,Device::CPU> const& A, Matrix<T,Device::CPU> const& B,
    T beta, Matrix<T>& C)
{
    EL_DEBUG_CSE;
    TrrkInternal(uplo, orientA, orientB, alpha, A, B, beta, C);
}

template<typename T>
void Trrk(
    UpperOrLower uplo,
    Orientation orientA, Orientation orientB,
    T alpha, Matrix<T,Device::CPU> const& A, Matrix<T,Device::CPU> const& B,
    T beta, Matrix<T>& C)
{
    EL_DEBUG_CSE;
    TrrkHelper(uplo, orientA, orientB, alpha, A, B, beta, C);
}

template<typename T>
void Trrk(
    UpperOrLower uplo, Orientation orientA, Orientation orientB,
    T alpha, AbstractDistMatrix<T> const& A, AbstractDistMatrix<T> const& B,
    T beta, AbstractDistMatrix<T>& C)
{
    EL_DEBUG_CSE;
    ScaleTrapezoid(beta, uplo, C);
    if (orientA==NORMAL && orientB==NORMAL)
        trrk::TrrkNN(uplo, alpha, A, B, C);
    else if (orientA==NORMAL)
        trrk::TrrkNT(uplo, orientB, alpha, A, B, C);
    else if (orientB==NORMAL)
        trrk::TrrkTN(uplo, orientA, alpha, A, B, C);
    else
        trrk::TrrkTT(uplo, orientA, orientB, alpha, A, B, C);
}

#define LOCALTRRK_PROTO_DEVICE(T,D) \
        template void LocalTrrk(                            \
        UpperOrLower uplo, Orientation orientA,         \
        T alpha,                                        \
        DistMatrix<T,STAR,MC,ELEMENT,D> const& A,        \
        DistMatrix<T,STAR,MR,ELEMENT,D> const& B,         \
        T beta, DistMatrix<T,MC,MR,ELEMENT,D>& C)

#define PROTO(T)                                        \
    template void Trrk(                                 \
        UpperOrLower uplo,                              \
        Orientation orientA, Orientation orientB,       \
        T alpha,                                        \
        Matrix<T,Device::CPU> const& A,                 \
        Matrix<T,Device::CPU> const& B,                 \
        T beta, Matrix<T>& C);                          \
    template void Trrk(                                 \
        UpperOrLower uplo,                              \
        Orientation orientA, Orientation orientB,       \
        T alpha,                                        \
        AbstractDistMatrix<T> const& A,                 \
        AbstractDistMatrix<T> const& B,                 \
        T beta, AbstractDistMatrix<T>& C);              \
    template void LocalTrrk(                            \
        UpperOrLower uplo,                              \
        T alpha,                                        \
        DistMatrix<T,MC,  STAR> const& A,               \
        DistMatrix<T,STAR,MR  > const& B,               \
        T beta, DistMatrix<T>& C);                      \
    template void LocalTrrk(                            \
        UpperOrLower uplo, Orientation orientB,         \
        T alpha,                                        \
        DistMatrix<T,MC,STAR> const& A,                 \
        DistMatrix<T,MR,STAR> const& B,                 \
        T beta, DistMatrix<T>& C);                      \
    template void LocalTrrk(                            \
        UpperOrLower uplo,                              \
        Orientation orientA, Orientation orientB,       \
        T alpha,                                        \
        DistMatrix<T,STAR,MC  > const& A,               \
        DistMatrix<T,MR,  STAR> const& B,               \
        T beta, DistMatrix<T>& C);                      \
    LOCALTRRK_PROTO_DEVICE(T, Device::CPU);

#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
#define EL_ENABLE_HALF
#include <El/macros/Instantiate.h>

#ifdef HYDROGEN_HAVE_GPU
#define LOCALTRRK_PROTO(T)                      \
    LOCALTRRK_PROTO_DEVICE(T, Device::GPU)
LOCALTRRK_PROTO(float);
LOCALTRRK_PROTO(double);
LOCALTRRK_PROTO(El::Complex<float>);
LOCALTRRK_PROTO(El::Complex<double>);
#endif // HYDROGEN_HAVE_GPU

} // namespace El
