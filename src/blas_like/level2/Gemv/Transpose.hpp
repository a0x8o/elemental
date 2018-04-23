/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/

namespace El {
namespace gemv {

template<Device D,typename T, typename=EnableIf<IsDeviceValidType<T,D>>>
void Transpose_impl
( Orientation orientation,
  T alpha,
  const AbstractDistMatrix<T>& APre,
  const AbstractDistMatrix<T>& x,
  T beta,
        AbstractDistMatrix<T>& yPre )
{
    EL_DEBUG_CSE
#ifndef EL_RELEASE
        AssertSameGrids( APre, x, yPre );
    if( ( x.Width() != 1 && x.Height() != 1 ) ||
        ( yPre.Width() != 1 && yPre.Height() != 1 )   )
        LogicError("Expected x and y to be vectors");
    const Int xLength = ( x.Width()==1 ? x.Height() : x.Width() );
    const Int yLength = ( yPre.Width()==1 ? yPre.Height() : yPre.Width() );
    if( APre.Height() != xLength || APre.Width() != yLength )
        LogicError
            ("Nonconformal: \n",DimsString(APre,"A"),"\n",
             DimsString(x,"x"),"\n",DimsString(yPre,"y"));
#endif // !EL_RELEASE
    const Grid& g = APre.Grid();

    DistMatrixReadProxy<T,T,MC,MR,ELEMENT,D> AProx( APre );
    DistMatrixReadWriteProxy<T,T,MC,MR,ELEMENT,D> yProx( yPre );
    auto& A = AProx.GetLocked();
    auto& y = yProx.Get();

    Scale( beta, y );
    if( x.Width() == 1 && y.Width() == 1 )
    {
        DistMatrix<T,MC,STAR,ELEMENT,D> x_MC_STAR(g);
        x_MC_STAR.AlignWith( A );
        x_MC_STAR = x;

        DistMatrix<T,MR,STAR,ELEMENT,D> z_MR_STAR(g);
        z_MR_STAR.AlignWith( A );
        z_MR_STAR.Resize( A.Width(), 1 );
        Zero( z_MR_STAR );
        LocalGemv( orientation, alpha, A, x_MC_STAR, T(0), z_MR_STAR );

        DistMatrix<T,MR,MC,ELEMENT,D> z_MR_MC(g);
        z_MR_MC.AlignWith( y );
        Contract( z_MR_STAR, z_MR_MC );
        Axpy( T(1), z_MR_MC, y );
    }
    else if( x.Width() == 1 )
    {
        DistMatrix<T,MC,STAR,ELEMENT,D> x_MC_STAR(g);
        x_MC_STAR.AlignWith( A );
        x_MC_STAR = x;

        DistMatrix<T,MR,STAR,ELEMENT,D> z_MR_STAR(g);
        z_MR_STAR.AlignWith( A );
        z_MR_STAR.Resize( A.Width(), 1 );
        Zero( z_MR_STAR );
        LocalGemv( orientation, alpha, A, x_MC_STAR, T(0), z_MR_STAR );

        DistMatrix<T,MR,MC,ELEMENT,D> z_MR_MC(g);
        z_MR_MC.AlignWith( y );
        Contract( z_MR_STAR, z_MR_MC );

        DistMatrix<T,MC,MR,ELEMENT,D> zTrans(g);
        zTrans.AlignWith( y );
        Transpose( z_MR_MC, zTrans );
        Axpy( T(1), zTrans, y );
    }
    else if( y.Width() == 1 )
    {
        DistMatrix<T,STAR,MC,ELEMENT,D> x_STAR_MC(g);
        x_STAR_MC.AlignWith( A );
        x_STAR_MC = x;

        DistMatrix<T,MR,STAR,ELEMENT,D> z_MR_STAR(g);
        z_MR_STAR.AlignWith( A );
        z_MR_STAR.Resize( A.Width(), 1 );
        Zero( z_MR_STAR );
        LocalGemv( orientation, alpha, A, x_STAR_MC, T(0), z_MR_STAR );

        DistMatrix<T,MR,MC,ELEMENT,D> z_MR_MC(g);
        z_MR_MC.AlignWith( y );
        Contract( z_MR_STAR, z_MR_MC );
        Axpy( T(1), z_MR_MC, y );
    }
    else
    {
        DistMatrix<T,STAR,MC,ELEMENT,D> x_STAR_MC(g);
        x_STAR_MC.AlignWith( A );
        x_STAR_MC = x;

        DistMatrix<T,MR,STAR,ELEMENT,D> z_MR_STAR(g);
        z_MR_STAR.AlignWith( A );
        z_MR_STAR.Resize( A.Width(), 1 );
        Zero( z_MR_STAR );
        LocalGemv( orientation, alpha, A, x_STAR_MC, T(0), z_MR_STAR );

        DistMatrix<T,MR,MC,ELEMENT,D> z_MR_MC(g);
        z_MR_MC.AlignWith( y );
        Contract( z_MR_STAR, z_MR_MC );

        DistMatrix<T,MC,MR,ELEMENT,D> zTrans(g);
        zTrans.AlignWith( y );
        Transpose( z_MR_MC, zTrans );
        Axpy( T(1), zTrans, y );
    }
}

template <Device D, typename T,
          typename=DisableIf<IsDeviceValidType<T,D>>,
          typename=void>
void Transpose_impl(
    Orientation,
    T,
    const AbstractDistMatrix<T>&,
    const AbstractDistMatrix<T>&,
    T,
    AbstractDistMatrix<T>& )
{
    LogicError("Gemv::Transpose_impl: type-device combo not supported.");
}

template<typename T>
void Transpose
( Orientation orientation,
  T alpha,
  const AbstractDistMatrix<T>& APre,
  const AbstractDistMatrix<T>& x,
  T beta,
        AbstractDistMatrix<T>& yPre )
{
    EL_DEBUG_CSE

    switch (APre.GetLocalDevice())
    {
    case Device::CPU:
        Transpose_impl<Device::CPU>(orientation, alpha, APre, x, beta, yPre);
        break;
#ifdef HYDROGEN_HAVE_CUDA
    case Device::GPU:
        Transpose_impl<Device::GPU>(orientation, alpha, APre, x, beta, yPre);
        break;
#endif // HYDROGEN_HAVE_CUDA
    default:
        LogicError("Gemv::Transpose: Bad device.");
    }

}

template<typename T,Device D>
void Transpose
( Orientation orientation,
  T alpha,
  const DistMatrix<T,MC,MR,ELEMENT,D>& A,
  const AbstractDistMatrix<T>& x,
  T beta,
  DistMatrix<T,VC,STAR,ELEMENT,D>& y )
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(
      AssertSameGrids( A, x, y );
      if( x.Width() != 1 || y.Width() != 1 )
          LogicError("Expected x and y to be column vectors");
      if( A.Height() != x.Height() || A.Width() != y.Height() )
          LogicError
          ("Nonconformal: \n",DimsString(A,"A"),"\n",
           DimsString(x,"x"),"\n",DimsString(y,"y"));
    )
    const Grid& g = A.Grid();
    Scale( beta, y );

    DistMatrix<T,MC,STAR> x_MC_STAR(g);
    x_MC_STAR.AlignWith( A );
    x_MC_STAR = x;

    DistMatrix<T,MR,STAR> z_MR_STAR(g);
    z_MR_STAR.AlignWith( A );
    z_MR_STAR.Resize( A.Width(), 1 );
    Zero( z_MR_STAR );
    LocalGemv( orientation, alpha, A, x_MC_STAR, T(0), z_MR_STAR );

    DistMatrix<T,VR,STAR> z_VR_STAR(g);
    z_VR_STAR.AlignWith( A );
    Contract( z_MR_STAR, z_VR_STAR );
    Axpy( T(1), z_VR_STAR, y );
}

} // namespace gemv
} // namespace El
