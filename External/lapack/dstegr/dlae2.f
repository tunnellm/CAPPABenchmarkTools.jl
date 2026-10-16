*> \brief \b DLAE2 computes the eigenvalues of a 2-by-2 symmetric matrix.
*
*  =========== DOCUMENTATION ===========
*
* Online html documentation available at
*            http://www.netlib.org/lapack/explore-html/
*
*> \htmlonly
*> Download DLAE2 + dependencies
*> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlae2.f">
*> [TGZ]</a>
*> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlae2.f">
*> [ZIP]</a>
*> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlae2.f">
*> [TXT]</a>
*> \endhtmlonly
*
*  Definition:
*  ===========
*
*       SUBROUTINE DLAE2( A, B, C, RT1, RT2, FLOPS )
*
*       .. Scalar Arguments ..
*       DOUBLE PRECISION   A, B, C, RT1, RT2
*       INTEGER*8          FLOPS
*       ..
*
*
*> \par Purpose:
*  =============
*>
*> \verbatim
*>
*> DLAE2  computes the eigenvalues of a 2-by-2 symmetric matrix
*>    [  A   B  ]
*>    [  B   C  ].
*> On return, RT1 is the eigenvalue of larger absolute value, and RT2
*> is the eigenvalue of smaller absolute value.
*> \endverbatim
*
*  Arguments:
*  ==========
*
*> \param[in] A
*> \verbatim
*>          A is DOUBLE PRECISION
*>          The (1,1) element of the 2-by-2 matrix.
*> \endverbatim
*>
*> \param[in] B
*> \verbatim
*>          B is DOUBLE PRECISION
*>          The (1,2) and (2,1) elements of the 2-by-2 matrix.
*> \endverbatim
*>
*> \param[in] C
*> \verbatim
*>          C is DOUBLE PRECISION
*>          The (2,2) element of the 2-by-2 matrix.
*> \endverbatim
*>
*> \param[out] RT1
*> \verbatim
*>          RT1 is DOUBLE PRECISION
*>          The eigenvalue of larger absolute value.
*> \endverbatim
*>
*> \param[out] RT2
*> \verbatim
*>          RT2 is DOUBLE PRECISION
*>          The eigenvalue of smaller absolute value.
*> \endverbatim
*
*  Authors:
*  ========
*
*> \author Univ. of Tennessee
*> \author Univ. of California Berkeley
*> \author Univ. of Colorado Denver
*> \author NAG Ltd.
*
*> \ingroup OTHERauxiliary
*
*> \par Further Details:
*  =====================
*>
*> \verbatim
*>
*>  RT1 is accurate to a few ulps barring over/underflow.
*>
*>  RT2 may be inaccurate if there is massive cancellation in the
*>  determinant A*C-B*B; higher precision or correctly rounded or
*>  correctly truncated arithmetic would be needed to compute RT2
*>  accurately in all cases.
*>
*>  Overflow is possible only if RT1 is within a factor of 5 of overflow.
*>  Underflow is harmless if the input data is 0 or exceeds
*>     underflow_threshold / macheps.
*> \endverbatim
*>
*  =====================================================================
      SUBROUTINE DLAE2( A, B, C, RT1, RT2, FLOPS )
*
*  -- LAPACK auxiliary routine --
*  -- LAPACK is a software package provided by Univ. of Tennessee,    --
*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
*
*     .. Scalar Arguments ..
      DOUBLE PRECISION   A, B, C, RT1, RT2
      INTEGER*8          FLOPS
*     ..
*
* =====================================================================
*
*     .. Parameters ..
      DOUBLE PRECISION   ONE
      PARAMETER          ( ONE = 1.0D0 )
      DOUBLE PRECISION   TWO
      PARAMETER          ( TWO = 2.0D0 )
      DOUBLE PRECISION   ZERO
      PARAMETER          ( ZERO = 0.0D0 )
      DOUBLE PRECISION   HALF
      PARAMETER          ( HALF = 0.5D0 )
*     ..
*     .. Local Scalars ..
      DOUBLE PRECISION   AB, ACMN, ACMX, ADF, DF, RT, SM, TB
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          ABS, SQRT
*     ..
*     .. Executable Statements ..
*
*     Compute the eigenvalues
*
      SM = A + C
      DF = A - C
      ADF = ABS( DF )
      TB = B + B
      AB = ABS( TB )
*     3 flops: A+C, A-C, B+B (ABS is free)
      FLOPS = FLOPS + 3
      IF( ABS( A ).GT.ABS( C ) ) THEN
         ACMX = A
         ACMN = C
      ELSE
         ACMX = C
         ACMN = A
      END IF
      IF( ADF.GT.AB ) THEN
*        4 flops: AB/ADF=1 DIV, ONE+(..)**2=1 FMA,
*        SQRT=1 SQRT, ADF*SQRT(..)=1 MUL
         RT = ADF*SQRT( ONE+( AB / ADF )**2 )
         FLOPS = FLOPS + 4
      ELSE IF( ADF.LT.AB ) THEN
*        4 flops: ADF/AB=1 DIV, ONE+(..)**2=1 FMA,
*        SQRT=1 SQRT, AB*SQRT(..)=1 MUL
         RT = AB*SQRT( ONE+( ADF / AB )**2 )
         FLOPS = FLOPS + 4
      ELSE
*
*        Includes case AB=ADF=0
*
*        2 flops: SQRT(TWO)=1 SQRT, AB*(..)=1 MUL
         RT = AB*SQRT( TWO )
         FLOPS = FLOPS + 2
      END IF
      IF( SM.LT.ZERO ) THEN
*        RT1: SM-RT=1 SUB, HALF*(..)=1 MUL => 2 ops
         RT1 = HALF*( SM-RT )
*
*        Order of execution important.
*        To get fully accurate smaller eigenvalue,
*        next line needs to be executed in higher precision.
*
*        RT2: ACMX/RT1=1 DIV, B/RT1=1 DIV,
*        (ACMX/RT1)*ACMN - (B/RT1)*B = A*B-C*D pattern = 1 MUL + 1 FMA
*        => 4 ops.  Total this branch: 6
         RT2 = ( ACMX / RT1 )*ACMN - ( B / RT1 )*B
         FLOPS = FLOPS + 6
      ELSE IF( SM.GT.ZERO ) THEN
*        RT1: SM+RT=1 ADD, HALF*(..)=1 MUL => 2 ops
         RT1 = HALF*( SM+RT )
*
*        Order of execution important.
*        To get fully accurate smaller eigenvalue,
*        next line needs to be executed in higher precision.
*
*        RT2: same as above => 4 ops.  Total this branch: 6
         RT2 = ( ACMX / RT1 )*ACMN - ( B / RT1 )*B
         FLOPS = FLOPS + 6
      ELSE
*
*        Includes case RT1 = RT2 = 0
*
*        2 flops: HALF*RT=1 MUL, -HALF*RT=1 MUL
         RT1 = HALF*RT
         RT2 = -HALF*RT
         FLOPS = FLOPS + 2
      END IF
      RETURN
*
*     End of DLAE2
*
      END
