      SUBROUTINE DLASSQ( N, X, INCX, SCALE, SUMSQ, FLOPS )
*
*  -- LAPACK auxiliary routine --
*     Update SCALE and SUMSQ such that
*        SCALE**2 * SUMSQ = sum(X(i)**2) + SCALE_in**2 * SUMSQ_in
*
*     .. Scalar Arguments ..
      INTEGER            INCX, N
      INTEGER*8          FLOPS
      DOUBLE PRECISION   SCALE, SUMSQ
*     ..
*     .. Array Arguments ..
      DOUBLE PRECISION   X( * )
*     ..
*
*     .. Local Scalars ..
      INTEGER            IX
      DOUBLE PRECISION   ABSXI
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          ABS
*     ..
      IF( N.GT.0 ) THEN
         DO IX = 1, 1 + ( N-1 )*INCX, INCX
            ABSXI = ABS( X( IX ) )
            IF( ABSXI.GT.SCALE ) THEN
*              SCALE/ABSXI = 1 DIV, (...)^2 = 1 MUL,
*              SUMSQ*(...)^2 + 1.0 = 1 FMA => 3 ops
               SUMSQ = 1.0D0 + SUMSQ*( SCALE / ABSXI )**2
               SCALE = ABSXI
               FLOPS = FLOPS + 3
            ELSE
*              ABSXI/SCALE = 1 DIV,
*              (ABSXI/SCALE)*(ABSXI/SCALE) + SUMSQ = 1 FMA => 2 ops
               SUMSQ = SUMSQ + ( ABSXI / SCALE )**2
               FLOPS = FLOPS + 2
            END IF
         END DO
      END IF
      RETURN
      END
