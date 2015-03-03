/*
   Copyright (c) 2009-2015, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "El.hpp"

namespace El {

template<typename T>
void AdjointAxpyContract
( T alpha, const AbstractDistMatrix<T>& A, 
                 AbstractDistMatrix<T>& B )
{
    DEBUG_ONLY(CallStackEntry cse("AdjointAxpyContract"))
    TransposeAxpyContract( alpha, A, B, true );
}

template<typename T>
void AdjointAxpyContract
( T alpha, const AbstractBlockDistMatrix<T>& A, 
                 AbstractBlockDistMatrix<T>& B )
{
    DEBUG_ONLY(CallStackEntry cse("AdjointAxpyContract"))
    TransposeAxpyContract( alpha, A, B, true );
}

#define PROTO(T) \
  template void AdjointAxpyContract \
  ( T alpha, const AbstractDistMatrix<T>& A, \
                   AbstractDistMatrix<T>& B ); \
  template void AdjointAxpyContract \
  ( T alpha, const AbstractBlockDistMatrix<T>& A, \
                   AbstractBlockDistMatrix<T>& B );

#define EL_ENABLE_QUAD
#include "El/macros/Instantiate.h"

} // namespace El
