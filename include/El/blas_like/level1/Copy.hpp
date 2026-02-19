/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_BLAS_COPY_HPP
#define EL_BLAS_COPY_HPP

#ifdef _OPENMP
#include <omp.h>
#endif

#include <El/hydrogen_config.h>

#include <El/core/Grid.hpp>
#include <El/blas_like/level1/Copy/internal_decl.hpp>
#include <El/blas_like/level1/Copy/GeneralPurpose.hpp>
#include <El/blas_like/level1/Copy/util.hpp>

<<<<<<< HEAD
#ifdef HYDROGEN_HAVE_GPU
#include <hydrogen/device/gpu/BasicCopy.hpp>
=======
namespace El {

template<typename T>
void Copy( const Matrix<T>& A, Matrix<T>& B )
{
    EL_DEBUG_CSE
    const Int height = A.Height();
    const Int width = A.Width();
    const Int size = height * width;
    B.Resize( height, width );
    const Int ldA = A.LDim();
    const Int ldB = B.LDim();
    const T* EL_RESTRICT ABuf = A.LockedBuffer();
          T* EL_RESTRICT BBuf = B.Buffer();

    if( ldA == height && ldB == height )
    {
#ifdef EL_HYBRID
        #pragma omp parallel
        {
            const Int numThreads = omp_get_num_threads();
            const Int thread = omp_get_thread_num();
            const Int chunk = (size + numThreads - 1) / numThreads;
            const Int start = Min(chunk * thread, size);
            const Int end = Min(chunk * (thread + 1), size);
            MemCopy( &BBuf[start], &ABuf[start], end - start );
        }
#else
        MemCopy( BBuf, ABuf, size );
#endif
    }
    else
    {
        EL_PARALLEL_FOR
        for( Int j=0; j<width; ++j )
        {
            MemCopy(&BBuf[j*ldB], &ABuf[j*ldA], height);
        }
    }

}

template<typename S,typename T,
         typename/*=EnableIf<CanCast<S,T>>*/>
void Copy( const Matrix<S>& A, Matrix<T>& B )
{
    EL_DEBUG_CSE
    EntrywiseMap( A, B, MakeFunction(Caster<S,T>::Cast) );
}

template<typename T,Dist U,Dist V>
void Copy( const ElementalMatrix<T>& A, DistMatrix<T,U,V>& B )
{
    EL_DEBUG_CSE
    B = A;
}

// Datatype conversions should not be very common, and so it is likely best to
// avoid explicitly instantiating every combination
template<typename S,typename T,Dist U,Dist V>
void Copy( const ElementalMatrix<S>& A, DistMatrix<T,U,V>& B )
{
    EL_DEBUG_CSE
    if( A.Grid() == B.Grid() && A.ColDist() == U && A.RowDist() == V )
    {
        if( !B.RootConstrained() )
            B.SetRoot( A.Root() );
        if( !B.ColConstrained() )
            B.AlignCols( A.ColAlign() );
        if( !B.RowConstrained() )
            B.AlignRows( A.RowAlign() );
        if( A.Root() == B.Root() &&
            A.ColAlign() == B.ColAlign() && A.RowAlign() == B.RowAlign() )
        {
            B.Resize( A.Height(), A.Width() );
            Copy( A.LockedMatrix(), B.Matrix() );
            return;
        }
    }
    DistMatrix<S,U,V> BOrig(A.Grid());
    BOrig.AlignWith( B );
    BOrig = A;
    B.Resize( A.Height(), A.Width() );
    Copy( BOrig.LockedMatrix(), B.Matrix() );
}

template<typename T,Dist U,Dist V>
void Copy( const BlockMatrix<T>& A, DistMatrix<T,U,V,BLOCK>& B )
{
    EL_DEBUG_CSE
    B = A;
}

// Datatype conversions should not be very common, and so it is likely best to
// avoid explicitly instantiating every combination
template<typename S,typename T,Dist U,Dist V>
void Copy( const BlockMatrix<S>& A, DistMatrix<T,U,V,BLOCK>& B )
{
    EL_DEBUG_CSE
    if( A.Grid() == B.Grid() && A.ColDist() == U && A.RowDist() == V )
    {
        if( !B.RootConstrained() )
            B.SetRoot( A.Root() );
        if( !B.ColConstrained() )
            B.AlignColsWith( A.DistData() );
        if( !B.RowConstrained() )
            B.AlignRowsWith( A.DistData() );
        if( A.Root() == B.Root() &&
            A.ColAlign() == B.ColAlign() &&
            A.RowAlign() == B.RowAlign() &&
            A.ColCut() == B.ColCut() &&
            A.RowCut() == B.RowCut() )
        {
            B.Resize( A.Height(), A.Width() );
            Copy( A.LockedMatrix(), B.Matrix() );
            return;
        }
    }
    DistMatrix<S,U,V,BLOCK> BOrig(A.Grid());
    BOrig.AlignWith( B );
    BOrig = A;
    B.Resize( A.Height(), A.Width() );
    Copy( BOrig.LockedMatrix(), B.Matrix() );
}

template<typename S,typename T,
         typename/*=EnableIf<CanCast<S,T>>*/>
void Copy( const ElementalMatrix<S>& A, ElementalMatrix<T>& B )
{
    EL_DEBUG_CSE
    #define GUARD(CDIST,RDIST,WRAP) \
      B.ColDist() == CDIST && B.RowDist() == RDIST && ELEMENT == WRAP
    #define PAYLOAD(CDIST,RDIST,WRAP) \
        auto& BCast = static_cast<DistMatrix<T,CDIST,RDIST,ELEMENT>&>(B); \
        Copy( A, BCast );
    #include <El/macros/GuardAndPayload.h>
}

template<typename T>
void Copy( const AbstractDistMatrix<T>& A, AbstractDistMatrix<T>& B )
{
    EL_DEBUG_CSE
    const DistWrap wrapA=A.Wrap(), wrapB=B.Wrap();
    if( wrapA == ELEMENT && wrapB == ELEMENT )
    {
        auto& ACast = static_cast<const ElementalMatrix<T>&>(A);
        auto& BCast = static_cast<ElementalMatrix<T>&>(B);
        Copy( ACast, BCast );
    }
    else if( wrapA == BLOCK && wrapB == BLOCK )
    {
        auto& ACast = static_cast<const BlockMatrix<T>&>(A);
        auto& BCast = static_cast<BlockMatrix<T>&>(B);
        Copy( ACast, BCast );
    }
    else
    {
        copy::GeneralPurpose( A, B );
    }
}

template<typename S,typename T,
         typename/*=EnableIf<CanCast<S,T>>*/>
void Copy( const AbstractDistMatrix<S>& A, AbstractDistMatrix<T>& B )
{
    EL_DEBUG_CSE
    const DistWrap wrapA=A.Wrap(), wrapB=B.Wrap();
    if( wrapA == ELEMENT && wrapB == ELEMENT )
    {
        auto& ACast = static_cast<const ElementalMatrix<S>&>(A);
        auto& BCast = static_cast<ElementalMatrix<T>&>(B);
        Copy( ACast, BCast );
    }
    else if( wrapA == BLOCK && wrapB == BLOCK )
    {
        auto& ACast = static_cast<const BlockMatrix<S>&>(A);
        auto& BCast = static_cast<BlockMatrix<T>&>(B);
        Copy( ACast, BCast );
    }
    else
    {
        copy::GeneralPurpose( A, B );
    }
}

template<typename S,typename T,
         typename/*=EnableIf<CanCast<S,T>>*/>
void Copy( const BlockMatrix<S>& A, BlockMatrix<T>& B )
{
    EL_DEBUG_CSE
    #define GUARD(CDIST,RDIST,WRAP) \
      B.ColDist() == CDIST && B.RowDist() == RDIST && BLOCK == WRAP
    #define PAYLOAD(CDIST,RDIST,WRAP) \
      auto& BCast = static_cast<DistMatrix<T,CDIST,RDIST,BLOCK>&>(B); \
      Copy( A, BCast );
    #include <El/macros/GuardAndPayload.h>
}

template<typename T>
void CopyFromRoot
( const Matrix<T>& A, DistMatrix<T,CIRC,CIRC>& B, bool includingViewers )
{
    EL_DEBUG_CSE
    if( B.CrossRank() != B.Root() )
        LogicError("Called CopyFromRoot from non-root");
    B.Resize( A.Height(), A.Width() );
    B.MakeSizeConsistent( includingViewers );
    B.Matrix() = A;
}

template<typename T>
void CopyFromNonRoot( DistMatrix<T,CIRC,CIRC>& B, bool includingViewers )
{
    EL_DEBUG_CSE
    if( B.CrossRank() == B.Root() )
        LogicError("Called CopyFromNonRoot from root");
    B.MakeSizeConsistent( includingViewers );
}

template<typename T>
void CopyFromRoot
( const Matrix<T>& A, DistMatrix<T,CIRC,CIRC,BLOCK>& B,
  bool includingViewers )
{
    EL_DEBUG_CSE
    if( B.CrossRank() != B.Root() )
        LogicError("Called CopyFromRoot from non-root");
    B.Resize( A.Height(), A.Width() );
    B.MakeSizeConsistent( includingViewers );
    B.Matrix() = A;
}

template<typename T>
void CopyFromNonRoot
( DistMatrix<T,CIRC,CIRC,BLOCK>& B, bool includingViewers )
{
    EL_DEBUG_CSE
    if( B.CrossRank() == B.Root() )
        LogicError("Called CopyFromNonRoot from root");
    B.MakeSizeConsistent( includingViewers );
}

template<typename T>
void Copy( const SparseMatrix<T>& A, SparseMatrix<T>& B )
{
    EL_DEBUG_CSE
    B = A;
}

template<typename S,typename T,
         typename/*=EnableIf<CanCast<S,T>>*/>
void Copy( const SparseMatrix<S>& A, SparseMatrix<T>& B )
{
    EL_DEBUG_CSE
    EntrywiseMap( A, B, MakeFunction(Caster<S,T>::Cast) );
}

template<typename S,typename T,
         typename/*=EnableIf<CanCast<S,T>>*/>
void Copy( const SparseMatrix<S>& A, Matrix<T>& B )
{
    EL_DEBUG_CSE
    const Int m = A.Height();
    const Int n = A.Width();
    const Int numEntries = A.NumEntries();
    const S* AValBuf = A.LockedValueBuffer();
    const Int* ARowBuf = A.LockedSourceBuffer();
    const Int* AColBuf = A.LockedTargetBuffer();

    T* BBuf = B.Buffer();
    const Int BLDim = B.LDim();

    B.Resize( m, n );
    Zero( B );
    for( Int e=0; e<numEntries; ++e )
        BBuf[ARowBuf[e]+AColBuf[e]*BLDim] = Caster<S,T>::Cast(AValBuf[e]);
}

template<typename T>
void Copy( const DistSparseMatrix<T>& A, DistSparseMatrix<T>& B )
{
    EL_DEBUG_CSE
    B = A;
}

template<typename S,typename T,
         typename/*=EnableIf<CanCast<S,T>>*/>
void Copy( const DistSparseMatrix<S>& A, DistSparseMatrix<T>& B )
{
    EL_DEBUG_CSE
    EntrywiseMap( A, B, MakeFunction(Caster<S,T>::Cast) );
}

template<typename S,typename T,
         typename/*=EnableIf<CanCast<S,T>>*/>
void Copy( const DistSparseMatrix<S>& A, AbstractDistMatrix<T>& B )
{
    EL_DEBUG_CSE
    const Int m = A.Height();
    const Int n = A.Width();
    const Int numEntries = A.NumLocalEntries();
    B.Resize( m, n );
    Zero( B );
    B.Reserve( numEntries );
    for( Int e=0; e<numEntries; ++e )
        B.QueueUpdate( A.Row(e), A.Col(e), Caster<S,T>::Cast(A.Value(e)) );
    B.ProcessQueues();
}

template<typename T>
void CopyFromRoot( const DistSparseMatrix<T>& ADist, SparseMatrix<T>& A )
{
    EL_DEBUG_CSE
    const Grid& grid = ADist.Grid();
    const int commSize = grid.Size();
    const int commRank = grid.Rank();

    const int numLocalEntries = ADist.NumLocalEntries();
    vector<int> entrySizes(commSize);
    mpi::AllGather( &numLocalEntries, 1, entrySizes.data(), 1, grid.Comm() );
    vector<int> entryOffs;
    const int numEntries = Scan( entrySizes, entryOffs );

    A.Resize( ADist.Height(), ADist.Width() );
    A.Reserve( numEntries );
    A.graph_.sources_.resize( numEntries );
    A.graph_.targets_.resize( numEntries );
    A.vals_.resize( numEntries );
    mpi::Gather
    ( ADist.LockedSourceBuffer(), numLocalEntries,
      A.SourceBuffer(), entrySizes.data(), entryOffs.data(),
      commRank, grid.Comm() );
    mpi::Gather
    ( ADist.LockedTargetBuffer(), numLocalEntries,
      A.TargetBuffer(), entrySizes.data(), entryOffs.data(),
      commRank, grid.Comm() );
    mpi::Gather
    ( ADist.LockedValueBuffer(), numLocalEntries,
      A.ValueBuffer(), entrySizes.data(), entryOffs.data(),
      commRank, grid.Comm() );
    A.ProcessQueues();
}

template<typename T>
void CopyFromNonRoot( const DistSparseMatrix<T>& ADist, int root )
{
    EL_DEBUG_CSE
    const Grid& grid = ADist.Grid();
    const int commSize = grid.Size();
    const int commRank = grid.Rank();
    if( commRank == root )
        LogicError("Root called CopyFromNonRoot");

    const int numLocalEntries = ADist.NumLocalEntries();
    vector<int> entrySizes(commSize);
    mpi::AllGather( &numLocalEntries, 1, entrySizes.data(), 1, grid.Comm() );
    vector<int> entryOffs;
    Scan( entrySizes, entryOffs );

    mpi::Gather
    ( ADist.LockedSourceBuffer(), numLocalEntries,
      (Int*)0, entrySizes.data(), entryOffs.data(), root, grid.Comm() );
    mpi::Gather
    ( ADist.LockedTargetBuffer(), numLocalEntries,
      (Int*)0, entrySizes.data(), entryOffs.data(), root, grid.Comm() );
    mpi::Gather
    ( ADist.LockedValueBuffer(), numLocalEntries,
      (T*)0, entrySizes.data(), entryOffs.data(), root, grid.Comm() );
}

template<typename T>
void Copy( const DistMultiVec<T>& A, DistMultiVec<T>& B )
{
    EL_DEBUG_CSE
    B.SetGrid( A.Grid() );
    B.Resize( A.Height(), A.Width() );
    B.Matrix() = A.LockedMatrix();
}

template<typename S,typename T,
         typename/*=EnableIf<CanCast<S,T>>*/>
void Copy( const DistMultiVec<S>& A, DistMultiVec<T>& B )
{
    EL_DEBUG_CSE
    EntrywiseMap( A, B, MakeFunction(Caster<S,T>::Cast) );
}

template<typename T>
void Copy( const DistMultiVec<T>& A, AbstractDistMatrix<T>& B )
{
    EL_DEBUG_CSE
    const Int m = A.Height();
    const Int n = A.Width();
    const Int mLoc = A.LocalHeight();
    B.Resize( m, n );
    Zero( B );
    B.Reserve( mLoc*n );
    auto& ALoc = A.LockedMatrix();
    for( Int iLoc=0; iLoc<mLoc; ++iLoc )
    {
        const Int i = A.GlobalRow(iLoc);
        for( Int j=0; j<n; ++j )
            B.QueueUpdate( i, j, ALoc(iLoc,j) );
    }
    B.ProcessQueues();
}

template<typename T>
void Copy( const AbstractDistMatrix<T>& A, DistMultiVec<T>& B )
{
    EL_DEBUG_CSE
    const Int m = A.Height();
    const Int n = A.Width();
    const Int mLoc = A.LocalHeight();
    const Int nLoc = A.LocalWidth();
    B.SetGrid( A.Grid() );
    B.Resize( m, n );
    Zero( B );
    B.Reserve( mLoc*nLoc );
    auto& ALoc = A.LockedMatrix();
    for( Int iLoc=0; iLoc<mLoc; ++iLoc )
    {
        const Int i = A.GlobalRow(iLoc);
        for( Int jLoc=0; jLoc<nLoc; ++jLoc )
        {
            const Int j = A.GlobalCol(jLoc);
            B.QueueUpdate( i, j, ALoc(iLoc,jLoc) );
        }
    }
    B.ProcessQueues();
}

template<typename T>
void CopyFromRoot( const DistMultiVec<T>& XDist, Matrix<T>& X )
{
    EL_DEBUG_CSE
    const Int m = XDist.Height();
    const Int n = XDist.Width();
    X.Resize( m, n, Max(m,1) );
    if( Min(m,n) == 0 )
        return;

    const Grid& grid = XDist.Grid();
    Output("grid.Size()=",grid.Size());
    const int commSize = grid.Size();
    const int commRank = grid.Rank();

    const int numLocalEntries = XDist.LocalHeight()*n;
    vector<int> entrySizes(commSize);
    mpi::AllGather( &numLocalEntries, 1, entrySizes.data(), 1, grid.Comm() );
    vector<int> entryOffs;
    const int numEntries = Scan( entrySizes, entryOffs );

    vector<T> recvBuf;
    FastResize( recvBuf, numEntries );

    const auto& XDistLoc = XDist.LockedMatrix();
    if( XDistLoc.Height() == XDistLoc.LDim() )
    {
        mpi::Gather
        ( XDistLoc.LockedBuffer(), numLocalEntries,
          recvBuf.data(), entrySizes.data(), entryOffs.data(),
          commRank, grid.Comm() );
    }
    else
    {
        vector<T> sendBuf;
        FastResize( sendBuf, numLocalEntries );
        for( Int jLoc=0; jLoc<XDistLoc.Width(); ++jLoc )
            for( Int iLoc=0; iLoc<XDistLoc.Height(); ++iLoc )
                sendBuf[iLoc+jLoc*XDistLoc.Height()] = XDistLoc(iLoc,jLoc);
        mpi::Gather
        ( sendBuf.data(), numLocalEntries,
          recvBuf.data(), entrySizes.data(), entryOffs.data(),
          commRank, grid.Comm() );
    }
    for( Int q=0; q<commSize; ++q )
    {
        const Int iOff = entryOffs[q]/n;
        const Int iSize = entrySizes[q]/n;
        for( Int t=0; t<entrySizes[q]; ++t )
            X( iOff+(t%iSize), t/iSize ) = recvBuf[entryOffs[q]+t];
    }
}

template<typename T>
void CopyFromNonRoot( const DistMultiVec<T>& XDist, int root )
{
    EL_DEBUG_CSE
    const Int m = XDist.Height();
    const Int n = XDist.Width();
    if( Min(m,n) == 0 )
        return;

    const Grid& grid = XDist.Grid();
    const int commSize = grid.Size();
    const int commRank = grid.Rank();
    if( commRank == root )
        LogicError("Called CopyFromNonRoot from root");

    const int numLocalEntries = XDist.LocalHeight()*XDist.Width();
    vector<int> entrySizes(commSize);
    mpi::AllGather( &numLocalEntries, 1, entrySizes.data(), 1, grid.Comm() );
    vector<int> entryOffs;
    Scan( entrySizes, entryOffs );

    const auto& XDistLoc = XDist.LockedMatrix();
    if( XDistLoc.Height() == XDistLoc.LDim() )
    {
        mpi::Gather
        ( XDistLoc.LockedBuffer(), numLocalEntries,
          (T*)0, entrySizes.data(), entryOffs.data(), root, grid.Comm() );
    }
    else
    {
        vector<T> sendBuf;
        FastResize( sendBuf, numLocalEntries );
        for( Int jLoc=0; jLoc<XDistLoc.Width(); ++jLoc )
            for( Int iLoc=0; iLoc<XDistLoc.Height(); ++iLoc )
                sendBuf[iLoc+jLoc*XDistLoc.Height()] = XDistLoc(iLoc,jLoc);
        mpi::Gather
        ( sendBuf.data(), numLocalEntries,
          (T*)0, entrySizes.data(), entryOffs.data(), root, grid.Comm() );
    }
}

#ifdef EL_INSTANTIATE_BLAS_LEVEL1
# define EL_EXTERN
#else
# define EL_EXTERN extern
>>>>>>> f46681a4a (Enable OpenMP code only if EL_HYBRID is set)
#endif

#include <hydrogen/meta/MetaUtilities.hpp>

// Introduce some metaprogramming notions.
//
// TODO: Move elsewhere.
namespace El
{

template <bool B>
using BoolVT = std::integral_constant<bool, B>;

namespace details
{

/** @brief A simple metafunction for interoping bitwise-equivalent
 *         types across device interfaces.
 */
template <typename T, Device D>
struct CompatibleStorageTypeT
{
    using type = T;
};

template <typename T, Device D>
using CompatibleStorageType = typename CompatibleStorageTypeT<T, D>::type;

#if defined(HYDROGEN_HAVE_HALF) && defined(HYDROGEN_GPU_USE_FP16)

template <>
struct CompatibleStorageTypeT<cpu_half_type, El::Device::GPU>
{
    using type = gpu_half_type;
};

template <>
struct CompatibleStorageTypeT<gpu_half_type, El::Device::CPU>
{
    using type = cpu_half_type;
};

#endif // defined(HYDROGEN_HAVE_HALF) && defined(HYDROGEN_GPU_USE_FP16)

template <typename T>
using CPUStorageType = CompatibleStorageType<T, Device::CPU>;

#ifdef HYDROGEN_HAVE_GPU
template <typename T>
using GPUStorageType = CompatibleStorageType<T, Device::GPU>;
#endif

// This layer of indirection checks the Tgt types and launches the
// copy if possible.
template <typename CopyFunctor,
          typename T, typename U, Device D1, Device D2,
          EnableWhen<IsStorageType<T, D1>, int> = 0>
void LaunchCopy(Matrix<T, D1> const& src, Matrix<U, D2>& tgt,
                CopyFunctor const& F)
{
   return F(src, tgt);
}

template <typename CopyFunctor,
          typename T, typename U, Device D1, Device D2,
          EnableUnless<IsStorageType<T, D1>, int> = 0>
void LaunchCopy(Matrix<T, D1> const&, Matrix<U, D2>&,
                CopyFunctor const&)
{
    LogicError("The combination U=", TypeTraits<U>::Name(), " "
               "and D=", DeviceName<D2>(), " is not supported.");
}

// This layer of indirection checks the Src types; this overload is
// also useful for some DistMatrix instantiations.
template <typename CopyFunctor,
          typename T, typename U, Device D2,
          EnableWhen<IsStorageType<U, D2>, int> = 0>
void LaunchCopy(AbstractMatrix<T> const& src, Matrix<U, D2>& tgt,
                CopyFunctor const& F)
{
    switch (src.GetDevice())
    {
    case Device::CPU:
        return LaunchCopy(
            static_cast<Matrix<T, Device::CPU> const&>(src), tgt, F);
#ifdef HYDROGEN_HAVE_GPU
    case Device::GPU:
        return LaunchCopy(
            static_cast<Matrix<T, Device::GPU> const&>(src), tgt, F);
#endif // HYDROGEN_HAVE_GPU
    default:
        LogicError("Copy: Bad device.");
    }
}

template <typename CopyFunctor,
          typename T, typename U, Device D2,
          EnableUnless<IsStorageType<U, D2>, int> = 0>
void LaunchCopy(AbstractMatrix<T> const&, Matrix<U, D2>&,
                CopyFunctor const&)
{
    LogicError("The combination U=", TypeTraits<U>::Name(), " "
               "and D=", DeviceName<D2>(), " is not supported.");
}

// The variadic templates allow these functors to be recycled across
// sequential and distributed matrices.

struct CopyFunctor
{
    template <typename... Args>
    void operator()(Args&&... args) const
    {
        return Copy(std::forward<Args>(args)...);
    }
};// CopyFunctor

struct CopyAsyncFunctor
{
    template <typename... Args>
    void operator()(Args&&... args) const
    {
        return CopyAsync(std::forward<Args>(args)...);
    }
};// CopyAsyncFunctor

}// namespace details
}// namespace El

//
// Include all the definitions
//
#include "CopyLocal.hpp"
#include "CopyAsyncLocal.hpp"
#include "CopyDistMatrix.hpp"
#include "CopyAsyncDistMatrix.hpp"
#include "CopyFromRoot.hpp"

namespace El
{

void Copy(BaseDistMatrix const&, BaseDistMatrix&);
void CopyAsync(BaseDistMatrix const&, BaseDistMatrix&);

}// namespace El

#endif // ifndef EL_BLAS_COPY_HPP
