/*
   Copyright (c) 2009-2012, Jack Poulson
   All rights reserved.

   This file is part of Elemental.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are met:

    - Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.

    - Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    - Neither the name of the owner nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
   ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
   LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
   CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
   SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
   INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
   CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
   ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
   POSSIBILITY OF SUCH DAMAGE.
*/

namespace elem {

namespace internal {

template<typename Real>
inline void
ExplicitQRHelper( Matrix<Real>& A )
{
    QR( A );
    ExpandPackedReflectors( LOWER, VERTICAL, 0, A );
}

template<typename Real>
inline void
ExplicitQRHelper( DistMatrix<Real>& A )
{
    QR( A );
    ExpandPackedReflectors( LOWER, VERTICAL, 0, A );
}

template<typename Real>
inline void
ExplicitQRHelper( Matrix<Complex<Real> >& A )
{
    Matrix<Complex<Real> > t;
    QR( A, t );
    ExpandPackedReflectors( LOWER, VERTICAL, UNCONJUGATED, 0, A, t );
}

template<typename Real>
inline void
ExplicitQRHelper( DistMatrix<Complex<Real> >& A )
{
    const Grid& g = A.Grid();
    DistMatrix<Complex<Real>,MD,STAR> t( g );
    QR( A, t );
    ExpandPackedReflectors( LOWER, VERTICAL, UNCONJUGATED, 0, A, t );
}

template<typename Real>
inline void
ExplicitQRHelper( Matrix<Real>& A, Matrix<Real>& R )
{
    QR( A );
    Matrix<Real> AT,
                 AB;
    PartitionDown
    ( A, AT,
         AB, std::min(A.Height(),A.Width()) );
    R = AT;
    MakeTrapezoidal( LEFT, UPPER, 0, R );
    ExpandPackedReflectors( LOWER, VERTICAL, 0, A );
}

template<typename Real>
inline void
ExplicitQRHelper( DistMatrix<Real>& A, DistMatrix<Real>& R )
{
    const Grid& g = A.Grid();
    QR( A );
    DistMatrix<Real> AT(g),
                     AB(g);
    PartitionDown
    ( A, AT,
         AB, std::min(A.Height(),A.Width()) );
    R = AT;
    MakeTrapezoidal( LEFT, UPPER, 0, R );
    ExpandPackedReflectors( LOWER, VERTICAL, 0, A );
}

template<typename Real>
inline void
ExplicitQRHelper( Matrix<Complex<Real> >& A, Matrix<Complex<Real> >& R )
{
    Matrix<Complex<Real> > t;
    QR( A, t );
    Matrix<Complex<Real> > AT,
                           AB;
    PartitionDown
    ( A, AT,
         AB, std::min(A.Height(),A.Width()) );
    R = AT;
    MakeTrapezoidal( LEFT, UPPER, 0, R );
    ExpandPackedReflectors( LOWER, VERTICAL, UNCONJUGATED, 0, A, t );
}

template<typename Real>
inline void
ExplicitQRHelper
( DistMatrix<Complex<Real> >& A, DistMatrix<Complex<Real> >& R )
{
    const Grid& g = A.Grid();
    DistMatrix<Complex<Real>,MD,STAR> t( g );
    QR( A, t );
    DistMatrix<Complex<Real> > AT(g),
                               AB(g);
    PartitionDown
    ( A, AT,
         AB, std::min(A.Height(),A.Width()) );
    R = AT;
    MakeTrapezoidal( LEFT, UPPER, 0, R );
    ExpandPackedReflectors( LOWER, VERTICAL, UNCONJUGATED, 0, A, t );
}

} // namespace internal

template<typename F> 
inline void
ExplicitQR( Matrix<F>& A )
{
#ifndef RELEASE
    PushCallStack("ExplicitQR");
#endif
    internal::ExplicitQRHelper( A );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename F> 
inline void
ExplicitQR( DistMatrix<F>& A )
{
#ifndef RELEASE
    PushCallStack("ExplicitQR");
#endif
    internal::ExplicitQRHelper( A );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename F> 
inline void
ExplicitQR( Matrix<F>& A, Matrix<F>& R )
{
#ifndef RELEASE
    PushCallStack("ExplicitQR");
#endif
    internal::ExplicitQRHelper( A, R );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename F> 
inline void
ExplicitQR( DistMatrix<F>& A, DistMatrix<F>& R )
{
#ifndef RELEASE
    PushCallStack("ExplicitQR");
#endif
    internal::ExplicitQRHelper( A, R );
#ifndef RELEASE
    PopCallStack();
#endif
}

} // namespace elem
