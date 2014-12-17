/*
   Copyright (c) 2009-2014, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "El.hpp"

namespace El {
namespace copy {

template<typename T,Dist U,Dist V>
void RowwiseVectorExchange
( const DistMatrix<T,STAR,ProductDist<U,V>()>& A,
        DistMatrix<T,STAR,ProductDist<V,U>()>& B )
{
    DEBUG_ONLY(CallStackEntry cse("copy::RowwiseVectorExchange"))
    AssertSameGrids( A, B );

    B.Resize( A.Height(), A.Width() );
    if( !B.Participating() )
        return;

    const Int height = B.Height();
    const Int localWidthA = A.LocalWidth();
    const Int localWidthB = B.LocalWidth();

    const Int sendSize = height*localWidthA;
    const Int recvSize = height*localWidthB;
    std::vector<T> buffer( sendSize+recvSize );
    T* sendBuf = &buffer[0];
    T* recvBuf = &buffer[sendSize];

    // Pack
    copy::util::InterleaveMatrix
    ( height, localWidthA,
      A.LockedBuffer(), 1, A.LDim(),
      sendBuf,          1, height );

    // Communicate
    const Int distSize = A.DistSize();
    const Int rowDiff = A.RowShift() - B.RowShift();
    const Int sendRankB = Mod( B.DistRank()+rowDiff, distSize );
    const Int recvRankA = Mod( A.DistRank()-rowDiff, distSize );
    const Int recvRankB =
      (recvRankA/A.PartialRowStride())+
      (recvRankA%A.PartialRowStride())*A.PartialUnionRowStride();
    mpi::SendRecv
    ( sendBuf, sendSize, sendRankB,
      recvBuf, recvSize, recvRankB, B.DistComm() );

    // Unpack
    copy::util::InterleaveMatrix
    ( height, localWidthB,
      recvBuf,    1, height,
      B.Buffer(), 1, B.LDim() );
}

#define PROTO_DIST(T,U,V) \
  template void RowwiseVectorExchange<T,U,V> \
  ( const DistMatrix<T,STAR,ProductDist<U,V>()>& A, \
          DistMatrix<T,STAR,ProductDist<V,U>()>& B );

#define PROTO(T) \
  PROTO_DIST(T,MC,MR) \
  PROTO_DIST(T,MR,MC)

#include "El/macros/Instantiate.h"

} // namespace copy
} // namespace El