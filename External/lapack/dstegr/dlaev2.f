*> \brief \b DLAEV2 computes the eigenvalues and eigenvectors of a 2-by-2 symmetric/Hermitian matrix.
*
*  =========== DOCUMENTATION ===========
*
* Online html documentation available at
*            http://www.netlib.org/lapack/explore-html/
*
*> \htmlonly
*> Download DLAEV2 + dependencies
*> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlaev2.f">
*> [TGZ]</a>
*> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlaev2.f">
*> [ZIP]</a>
*> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlaev2.f">
*> [TXT]</a>
*> \endhtmlonly
*
*  Definition:
*  ===========
*
*       SUBROUTINE DLAEV2( A, B, C, RT1, RT2, CS1, SN1, FLOPS )
*
*       .. Scalar Arguments ..
*       DOUBLE PRECISION   A, B, C, CS1, RT1, RT2, SN1
*       INTEGER*8          FLOPS
*       ..
*
*
*> \par Purpose:
*  =============
*>
*> \verbatim
*>
*> DLAEV2 computes the eigendecomposition of a 2-by-2 symmetric matrix
*>    [  A   B  ]
*>    [  B   C  ].
*> On return, RT1 is the eigenvalue of larger absolute value, RT2 is the
*> eigenvalue of smaller absolute value, and (CS1,SN1) is the unit right
*> eigenvector for RT1, giving the decomposition
*>
*>    [ CS1  SN1 ] [  A   B  ] [ CS1 -SN1 ]  =  [ RT1  0  ]
*>    [-SN1  CS1 ] [  B   C  ] [ SN1  CS1 ]     [  0  RT2 ].
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
*>          The (1,2) element and the conjugate of the (2,1) element of
*>          the 2-by-2 matrix.
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
*>
*> \param[out] CS1
*> \verbatim
*>          CS1 is DOUBLE PRECISION
*> \endverbatim
*>
*> \param[out] SN1
*> \verbatim
*>          SN1 is DOUBLE PRECISION
*>          The vector (CS1, SN1) is a unit right eigenvector for RT1.
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
*>  CS1 and SN1 are accurate to a few ulps barring over/underflow.
*>
*>  Overflow is possible only if RT1 is within a factor of 5 of overflow.
*>  Underflow is harmless if the input data is 0 or exceeds
*>     underflow_threshold / macheps.
*> \endverbatim
*>
*  =====================================================================
      SUBROUTINE DLAEV2( A, B, C, RT1, RT2, CS1, SN1, FLOPS )
*
*  -- LAPACK auxiliary routine --
*  -- LAPACK is a software package provided by Univ. of Tennessee,    --
*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
*
*     .. Scalar Arguments ..
      DOUBLE PRECISION   A, B, C, CS1, RT1, RT2, SN1
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
      INTEGER            SGN1, SGN2
      DOUBLE PRECISION   AB, ACMN, ACMX, ACS, ADF, CS, CT, DF, RT, SM,
     $                   TB, TN
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          ABS, SQRT
*     ..
*     .. Executable Statements ..
*
*     Compute the eigenvalues
*
*     SM = A + C: 1 ADD
*     DF = A - C: 1 SUB
*     TB = B + B: 1 ADD
*     3 ops total
      SM = A + C
      DF = A - C
      ADF = ABS( DF )
      TB = B + B
      AB = ABS( TB )
      FLOPS = FLOPS + 3
      IF( ABS( A ).GT.ABS( C ) ) THEN
         ACMX = A
         ACMN = C
      ELSE
         ACMX = C
         ACMN = A
      END IF
      IF( ADF.GT.AB ) THEN
*        RT = ADF*SQRT(ONE+(AB/ADF)**2)
*        AB/ADF = 1 DIV, ONE+(..)**2 = 1 FMA,
*        SQRT = 1 SQRT, ADF*SQRT(...) = 1 MUL => 4 ops
         RT = ADF*SQRT( ONE+( AB / ADF )**2 )
         FLOPS = FLOPS + 4
      ELSE IF( ADF.LT.AB ) THEN
*        Same structure as above => 4 ops
         RT = AB*SQRT( ONE+( ADF / AB )**2 )
         FLOPS = FLOPS + 4
      ELSE
*
*        Includes case AB=ADF=0
*        RT = AB*SQRT(TWO): SQRT = 1, MUL = 1 => 2 ops
         RT = AB*SQRT( TWO )
         FLOPS = FLOPS + 2
      END IF
      IF( SM.LT.ZERO ) THEN
*        RT1 = HALF*(SM-RT): 1 SUB + 1 MUL = 2 ops
*        RT2 = (ACMX/RT1)*ACMN - (B/RT1)*B:
*          2 DIV + pattern A*B - C*D = 1 MUL + 1 FMA = 2 ops => 4 ops
*        Total: 6 ops
         RT1 = HALF*( SM-RT )
         SGN1 = -1
*
*        Order of execution important.
*        To get fully accurate smaller eigenvalue,
*        next line needs to be executed in higher precision.
*
         RT2 = ( ACMX / RT1 )*ACMN - ( B / RT1 )*B
         FLOPS = FLOPS + 6
      ELSE IF( SM.GT.ZERO ) THEN
*        Same structure as SM < 0 branch => 6 ops
         RT1 = HALF*( SM+RT )
         SGN1 = 1
*
*        Order of execution important.
*        To get fully accurate smaller eigenvalue,
*        next line needs to be executed in higher precision.
*
         RT2 = ( ACMX / RT1 )*ACMN - ( B / RT1 )*B
         FLOPS = FLOPS + 6
      ELSE
*
*        Includes case RT1 = RT2 = 0
*        RT1 = HALF*RT: 1 MUL
*        RT2 = -HALF*RT: 1 MUL
*        Total: 2 ops
         RT1 = HALF*RT
         RT2 = -HALF*RT
         SGN1 = 1
         FLOPS = FLOPS + 2
      END IF
*
*     Compute the eigenvector
*
*     CS = DF +/- RT: 1 op
      IF( DF.GE.ZERO ) THEN
         CS = DF + RT
         SGN2 = 1
      ELSE
         CS = DF - RT
         SGN2 = -1
      END IF
      ACS = ABS( CS )
      FLOPS = FLOPS + 1
      IF( ACS.GT.AB ) THEN
*        CT = -TB / CS: 1 DIV
*        SN1 = ONE / SQRT(ONE+CT*CT):
*          CT*CT feeds into ONE+...: 1 FMA, SQRT: 1, ONE/...: 1 DIV
*          => 3 ops
*        CS1 = CT*SN1: 1 MUL
*        Total: 5 ops
         CT = -TB / CS
         SN1 = ONE / SQRT( ONE+CT*CT )
         CS1 = CT*SN1
         FLOPS = FLOPS + 5
      ELSE
         IF( AB.EQ.ZERO ) THEN
*           No arithmetic, just constant assignments: 0 ops
            CS1 = ONE
            SN1 = ZERO
         ELSE
*           TN = -CS / TB: 1 DIV
*           CS1 = ONE / SQRT(ONE+TN*TN): 1 FMA + 1 SQRT + 1 DIV = 3
*           SN1 = TN*CS1: 1 MUL
*           Total: 5 ops
            TN = -CS / TB
            CS1 = ONE / SQRT( ONE+TN*TN )
            SN1 = TN*CS1
            FLOPS = FLOPS + 5
         END IF
      END IF
      IF( SGN1.EQ.SGN2 ) THEN
         TN = CS1
         CS1 = -SN1
         SN1 = TN
      END IF
      RETURN
*
*     End of DLAEV2
*
      END
