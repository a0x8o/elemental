/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El-lite.hpp>
#include <El/blas_like/level1.hpp>
#include <El/blas_like/level2.hpp>
#include <El/blas_like/level3.hpp>

#include "./Trsm/LLN.hpp"
#include "./Trsm/LLT.hpp"
#include "./Trsm/LUN.hpp"
#include "./Trsm/LUT.hpp"
#include "./Trsm/RLN.hpp"
#include "./Trsm/RLT.hpp"
#include "./Trsm/RUN.hpp"
#include "./Trsm/RUT.hpp"
#include "core/environment/decl.hpp"

namespace El {

#ifdef HYDROGEN_HAVE_GPU
template<typename F>
void Trsm(
    LeftOrRight side,
    UpperOrLower uplo,
    Orientation orientation,
    UnitOrNonUnit diag,
    F alpha,
    Matrix<F, Device::GPU> const& A,
    Matrix<F, Device::GPU>& B,
    bool const)
{
    auto multisync = MakeMultiSync(
        SyncInfoFromMatrix(B), SyncInfoFromMatrix(A));

    gpu_blas::Trsm(
        LeftOrRightToSideMode(side), UpperOrLowerToFillMode(uplo),
        OrientationToTransposeMode(orientation),
        UnitOrNonUnitToDiagType(diag),
        B.Height(), B.Width(),
        alpha, A.LockedBuffer(), A.LDim(),
        B.Buffer(), B.LDim(),
        multisync);
}
#endif // HYDROGEN_HAVE_GPU

template<typename F>
void Trsm(
    LeftOrRight side,
    UpperOrLower uplo,
    Orientation orientation,
    UnitOrNonUnit diag,
    F alpha,
    Matrix<F> const& A,
    Matrix<F>& B,
    bool checkIfSingular)
{
    if (checkIfSingular && diag != UNIT)
    {
        const Int n = A.Height();
        for (Int j=0; j<n; ++j)
            if (A.Get(j,j) == F(0))
                throw SingularMatrixException();
    }
    const char sideChar = LeftOrRightToChar(side);
    const char uploChar = UpperOrLowerToChar(uplo);
    const char transChar = OrientationToChar(orientation);
    const char diagChar = UnitOrNonUnitToChar(diag);
    blas::Trsm(
        sideChar, uploChar, transChar, diagChar,
        B.Height(), B.Width(),
        alpha, A.LockedBuffer(), A.LDim(), B.Buffer(), B.LDim());
}

template <typename F>
void Trsm(
    LeftOrRight side,
    UpperOrLower uplo,
    Orientation orientation,
    UnitOrNonUnit diag,
    F alpha,
    AbstractMatrix<F> const& A,
    AbstractMatrix<F>& B,
    bool checkIfSingular)
{
    EL_DEBUG_CSE;
#ifndef EL_RELEASE
    if (A.Height() != A.Width())
        LogicError("Triangular matrix must be square");
    if (side == LEFT)
    {
        if (A.Height() != B.Height())
            LogicError("Nonconformal Trsm");
    }
    else
    {
        if (A.Height() != B.Width())
            LogicError("Nonconformal Trsm");
    }
#endif // EL_RELEASE

    switch (A.GetDevice())
    {
    case Device::CPU:
        Trsm(side, uplo, orientation, diag,
             alpha, static_cast<Matrix<F, Device::CPU> const&>(A),
             static_cast<Matrix<F, Device::CPU>&>(B),
             checkIfSingular);
        break;
#ifdef HYDROGEN_HAVE_GPU
    case Device::GPU:
        Trsm(side, uplo, orientation, diag,
             alpha, static_cast<Matrix<F, Device::GPU> const&>(A),
             static_cast<Matrix<F, Device::GPU>&>(B),
             checkIfSingular);
        break;
#endif
    default:
        RuntimeError("Unknown device.");
    }
}

// TODO: Make the TRSM_DEFAULT switching mechanism smarter (perhaps, empirical)
template <typename F, Device D>
void Trsm
( LeftOrRight side,
  UpperOrLower uplo,
  Orientation orientation,
  UnitOrNonUnit diag,
  F alpha,
  AbstractDistMatrix<F> const& A,
  AbstractDistMatrix<F>& B,
  bool checkIfSingular,
  TrsmAlgorithm alg,
  DeviceTag<D> dtag)
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(
      AssertSameGrids( A, B );
      if( A.Height() != A.Width() )
          LogicError("A must be square");
      if( side == LEFT )
      {
          if( A.Height() != B.Height() )
              LogicError("Nonconformal Trsm");
      }
      else
      {
          if( A.Height() != B.Width() )
              LogicError("Nonconformal Trsm");
      }
    )
    B *= alpha;

    // Call the single right-hand side algorithm if appropriate
    if( side == LEFT && B.Width() == 1 && D == Device::CPU )
    {
        Trsv( uplo, orientation, diag, A, B );
        return;
    }
    // TODO: Compute appropriate transpose/conjugation options to convert
    //       to Trsv.
    /*
    else if( side == RIGHT && B.Height() == 1 )
    {
        Trsv( uplo, orientation, diag, A, B );
        return;
    }
    */

    const Int p = B.Grid().Size();
    if( side == LEFT && uplo == LOWER )
    {
        if( orientation == NORMAL )
        {
            if( alg == TRSM_DEFAULT )
            {
                if( B.Width() > 5*p )
                    trsm::LLNLarge( diag, A, B, checkIfSingular, dtag );
                else
                    trsm::LLNMedium( diag, A, B, checkIfSingular, dtag );
            }
            else if( alg == TRSM_LARGE )
                trsm::LLNLarge( diag, A, B, checkIfSingular, dtag );
            else if( alg == TRSM_MEDIUM )
                trsm::LLNMedium( diag, A, B, checkIfSingular, dtag );
            else if( alg == TRSM_SMALL )
            {
                if( A.ColDist() == VR )
                {
                    DistMatrixReadProxy<F,F,VR,STAR,ELEMENT,D> AProx( A );
                    auto& APost = AProx.GetLocked();

                    ElementalProxyCtrl ctrl;
                    ctrl.colConstrain = true;
                    ctrl.colAlign = APost.ColAlign();

                    DistMatrixReadWriteProxy<F,F,VR,STAR,ELEMENT,D> BProx( B, ctrl );
                    auto& BPost = BProx.Get();

                    trsm::LLNSmall( diag, APost, BPost, checkIfSingular );
                }
                else
                {
                    DistMatrixReadProxy<F,F,VC,STAR,ELEMENT,D> AProx( A );
                    auto& APost = AProx.GetLocked();

                    ElementalProxyCtrl ctrl;
                    ctrl.colConstrain = true;
                    ctrl.colAlign = APost.ColAlign();

                    DistMatrixReadWriteProxy<F,F,VC,STAR,ELEMENT,D> BProx( B, ctrl );
                    auto& BPost = BProx.Get();

                    trsm::LLNSmall( diag, APost, BPost, checkIfSingular );
                }
            }
            else
                LogicError("Unsupported TRSM algorithm");
        }
        else
        {
            if( alg == TRSM_DEFAULT )
            {
                if( B.Width() > 5*p )
                    trsm::LLTLarge( orientation, diag, A, B, checkIfSingular, dtag );
                else
                    trsm::LLTMedium( orientation, diag, A, B, checkIfSingular, dtag );
            }
            else if( alg == TRSM_LARGE )
                trsm::LLTLarge( orientation, diag, A, B, checkIfSingular, dtag );
            else if( alg == TRSM_MEDIUM )
                trsm::LLTMedium( orientation, diag, A, B, checkIfSingular, dtag );
            else if( alg == TRSM_SMALL )
            {
                if( A.ColDist() == VR )
                {
                    DistMatrixReadProxy<F,F,VR,STAR,ELEMENT,D> AProx( A );
                    auto& APost = AProx.GetLocked();

                    ElementalProxyCtrl ctrl;
                    ctrl.colConstrain = true;
                    ctrl.colAlign = APost.ColAlign();

                    DistMatrixReadWriteProxy<F,F,VR,STAR,ELEMENT,D> BProx( B, ctrl );
                    auto& BPost = BProx.Get();

                    trsm::LLTSmall
                    ( orientation, diag, APost, BPost, checkIfSingular );
                }
                else if( A.RowDist() == VC )
                {
                    DistMatrixReadProxy<F,F,STAR,VC,ELEMENT,D> AProx( A );
                    auto& APost = AProx.GetLocked();

                    ElementalProxyCtrl ctrl;
                    ctrl.colConstrain = true;
                    ctrl.colAlign = APost.RowAlign();

                    DistMatrixReadWriteProxy<F,F,VC,STAR,ELEMENT,D> BProx( B, ctrl );
                    auto& BPost = BProx.Get();

                    trsm::LLTSmall
                    ( orientation, diag, APost, BPost, checkIfSingular );
                }
                else if( A.RowDist() == VR )
                {
                    DistMatrixReadProxy<F,F,STAR,VR,ELEMENT,D> AProx( A );
                    auto& APost = AProx.GetLocked();

                    ElementalProxyCtrl ctrl;
                    ctrl.colConstrain = true;
                    ctrl.colAlign = A.RowAlign();

                    DistMatrixReadWriteProxy<F,F,VR,STAR,ELEMENT,D> BProx( B, ctrl );
                    auto& BPost = BProx.Get();

                    trsm::LLTSmall
                    ( orientation, diag, APost, BPost, checkIfSingular );
                }
                else
                {
                    DistMatrixReadProxy<F,F,VC,STAR,ELEMENT,D> AProx( A );
                    auto& APost = AProx.GetLocked();

                    ElementalProxyCtrl ctrl;
                    ctrl.colConstrain = true;
                    ctrl.colAlign = A.ColAlign();

                    DistMatrixReadWriteProxy<F,F,VC,STAR,ELEMENT,D> BProx( B, ctrl );
                    auto& BPost = BProx.Get();

                    trsm::LLTSmall
                    ( orientation, diag, APost, BPost, checkIfSingular );
                }
            }
            else
                LogicError("Unsupported TRSM algorithm");
        }
    }
    else if( side == LEFT && uplo == UPPER )
    {
        if( orientation == NORMAL )
        {
            if( alg == TRSM_DEFAULT )
            {
                if( B.Width() > 5*p )
                    trsm::LUNLarge( diag, A, B, checkIfSingular, dtag );
                else
                    trsm::LUNMedium( diag, A, B, checkIfSingular, dtag );
            }
            else if( alg == TRSM_LARGE )
                trsm::LUNLarge( diag, A, B, checkIfSingular, dtag );
            else if( alg == TRSM_MEDIUM )
                trsm::LUNMedium( diag, A, B, checkIfSingular, dtag );
            else if( alg == TRSM_SMALL )
            {
                if( A.ColDist() == VR )
                {
                    DistMatrixReadProxy<F,F,VR,STAR,ELEMENT,D> AProx( A );
                    auto& APost = AProx.GetLocked();

                    ElementalProxyCtrl ctrl;
                    ctrl.colConstrain = true;
                    ctrl.colAlign = A.ColAlign();

                    DistMatrixReadWriteProxy<F,F,VR,STAR,ELEMENT,D> BProx( B, ctrl );
                    auto& BPost = BProx.Get();

                    trsm::LUNSmall( diag, APost, BPost, checkIfSingular );
                }
                else
                {
                    DistMatrixReadProxy<F,F,VC,STAR,ELEMENT,D> AProx( A );
                    auto& APost = AProx.GetLocked();

                    ElementalProxyCtrl ctrl;
                    ctrl.colConstrain = true;
                    ctrl.colAlign = A.ColAlign();

                    DistMatrixReadWriteProxy<F,F,VC,STAR,ELEMENT,D> BProx( B, ctrl );
                    auto& BPost = BProx.Get();

                    trsm::LUNSmall( diag, APost, BPost, checkIfSingular );
                }
            }
            else
                LogicError("Unsupported TRSM algorithm");
        }
        else
        {
            if( alg == TRSM_DEFAULT )
            {
                if( B.Width() > 5*p )
                    trsm::LUTLarge( orientation, diag, A, B, checkIfSingular, dtag );
                else
                    trsm::LUTMedium( orientation, diag, A, B, checkIfSingular, dtag );
            }
            else if( alg == TRSM_LARGE )
                trsm::LUTLarge( orientation, diag, A, B, checkIfSingular, dtag );
            else if( alg == TRSM_MEDIUM )
                trsm::LUTMedium( orientation, diag, A, B, checkIfSingular, dtag );
            else if( alg == TRSM_SMALL )
            {
                if( A.RowDist() == VC )
                {
                    DistMatrixReadProxy<F,F,STAR,VC,ELEMENT,D> AProx( A );
                    auto& APost = AProx.GetLocked();

                    ElementalProxyCtrl ctrl;
                    ctrl.colConstrain = true;
                    ctrl.colAlign = A.RowAlign();

                    DistMatrixReadWriteProxy<F,F,VC,STAR,ELEMENT,D> BProx( B, ctrl );
                    auto& BPost = BProx.Get();

                    trsm::LUTSmall
                    ( orientation, diag, APost, BPost, checkIfSingular );
                }
                else
                {
                    DistMatrixReadProxy<F,F,STAR,VR,ELEMENT,D> AProx( A );
                    auto& APost = AProx.GetLocked();

                    ElementalProxyCtrl ctrl;
                    ctrl.colConstrain = true;
                    ctrl.colAlign = A.RowAlign();

                    DistMatrixReadWriteProxy<F,F,VR,STAR,ELEMENT,D> BProx( B, ctrl );
                    auto& BPost = BProx.Get();

                    trsm::LUTSmall
                    ( orientation, diag, APost, BPost, checkIfSingular );
                }
            }
            else
                LogicError("Unsupported TRSM algorithm");
        }
    }
    else if( side == RIGHT && uplo == LOWER )
    {
        if( orientation == NORMAL )
        {
            if( alg == TRSM_DEFAULT )
                trsm::RLN( diag, A, B, checkIfSingular, dtag );
            else
                LogicError("Unsupported TRSM algorithm");
        }
        else
        {
            if( alg == TRSM_DEFAULT )
                trsm::RLT( orientation, diag, A, B, checkIfSingular, dtag );
            else
                LogicError("Unsupported TRSM algorithm");
        }
    }
    else if( side == RIGHT && uplo == UPPER )
    {
        if( orientation == NORMAL )
        {
            if( alg == TRSM_DEFAULT )
                trsm::RUN( diag, A, B, checkIfSingular, dtag );
            else
                LogicError("Unsupported TRSM algorithm");
        }
        else
        {
            if( alg == TRSM_DEFAULT )
                trsm::RUT( orientation, diag, A, B, checkIfSingular, dtag );
            else
                LogicError("Unsupported TRSM algorithm");
        }
    }
}

#if defined HYDROGEN_HAVE_HALF && defined HYDROGEN_HAVE_GPU
void Trsm(LeftOrRight,
          UpperOrLower,
          Orientation,
          UnitOrNonUnit,
          cpu_half_type,
          AbstractDistMatrix<cpu_half_type> const&,
          AbstractDistMatrix<cpu_half_type>&,
          bool,
          TrsmAlgorithm,
          DeviceTag<Device::GPU>)
{
  // Shouldn't get here
  RuntimeError("TRSM not supported for cpu_half_type on GPUs.");
}
#if defined HYDROGEN_GPU_USE_FP16
void Trsm(LeftOrRight,
          UpperOrLower,
          Orientation,
          UnitOrNonUnit,
          gpu_half_type,
          AbstractDistMatrix<gpu_half_type> const&,
          AbstractDistMatrix<gpu_half_type>&,
          bool,
          TrsmAlgorithm,
          DeviceTag<Device::CPU>)
{
  // Shouldn't get here
  RuntimeError("TRSM not supported for gpu_half_type on CPUs.");
}
#endif // defined HYDROGEN_GPU_USE_FP16
#endif // defined HYDROGEN_HAVE_HALF && defined HYDROGEN_HAVE_GPU

template <typename F>
void Trsm(LeftOrRight side,
          UpperOrLower uplo,
          Orientation orientation,
          UnitOrNonUnit diag,
          F alpha,
          AbstractDistMatrix<F> const& A,
          AbstractDistMatrix<F>& B,
          bool checkIfSingular,
          TrsmAlgorithm alg)
{
  switch (A.GetLocalDevice()) {
  case Device::CPU:
    Trsm(side, uplo, orientation, diag, alpha, A, B, checkIfSingular, alg,
         DeviceTag<Device::CPU>{});
    break;
#ifdef HYDROGEN_HAVE_GPU
  case Device::GPU:
    Trsm(side, uplo, orientation, diag, alpha, A, B, checkIfSingular, alg,
         DeviceTag<Device::GPU>{});
    break;
#endif // HYDROGEN_HAVE_GPU
  default:
    LogicError("Unknown device.");
  }
}

template<typename F, Device D>
void LocalTrsm
( LeftOrRight side,
  UpperOrLower uplo,
  Orientation orientation,
  UnitOrNonUnit diag,
  F alpha,
  DistMatrix<F,STAR,STAR,ELEMENT,D> const& A,
  AbstractDistMatrix<F>& X,
  bool checkIfSingular )
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(
      if( (side == LEFT && X.ColDist() != STAR) ||
          (side == RIGHT && X.RowDist() != STAR) )
          LogicError
          ("Dist of RHS must conform with that of triangle");
    )
    if (X.GetLocalDevice() != D)
        LogicError("LocalTrsm: Device mismatch.");
    Trsm(side,
         uplo,
         orientation,
         diag,
         alpha,
         A.LockedMatrix(),
         static_cast<Matrix<F, D>&>(X.Matrix()),
         checkIfSingular);
}


#define LOCALTRSM_PROTO_DEVICE(F, D)                                           \
    template void LocalTrsm(LeftOrRight side,                                  \
                            UpperOrLower uplo,                                 \
                            Orientation orientation,                           \
                            UnitOrNonUnit diag,                                \
                            F alpha,                                           \
                            DistMatrix<F, STAR, STAR, ELEMENT, D> const& A,    \
                            AbstractDistMatrix<F>& X,                          \
                            bool checkIfSingular)
#ifdef HYDROGEN_HAVE_GPU
#define LOCALTRSM_PROTO(F)                                                     \
    LOCALTRSM_PROTO_DEVICE(F, Device::CPU);                                    \
    LOCALTRSM_PROTO_DEVICE(F, Device::GPU)
#else
#define LOCALTRSM_PROTO(F)                                                     \
    LOCALTRSM_PROTO_DEVICE(F, Device::CPU)
#endif

#define PROTO(F) \
  template void Trsm \
  ( LeftOrRight side, \
    UpperOrLower uplo, \
    Orientation orientation, \
    UnitOrNonUnit diag, \
    F alpha, \
    AbstractMatrix<F> const& A, \
    AbstractMatrix<F>& B,       \
    bool checkIfSingular ); \
  template void Trsm \
  ( LeftOrRight side, \
    UpperOrLower uplo, \
    Orientation orientation, \
    UnitOrNonUnit diag, \
    F alpha, \
    const AbstractDistMatrix<F>& A, \
          AbstractDistMatrix<F>& B, \
    bool checkIfSingular, \
    TrsmAlgorithm alg ); \
  LOCALTRSM_PROTO(F);

#define EL_NO_INT_PROTO
#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGFLOAT
#define EL_ENABLE_HALF
#include <El/macros/Instantiate.h>

} // namespace El
