C  Modified from LAPACK's DSTERF (Pal-Walker-Kahan QL/QR variant).
C  Original: Copyright (c) 1992-2024 The University of Tennessee.
C            All rights reserved.
C  Licensed under the 3-clause BSD license (see LICENSE).
C  Modifications by CAPPABenchmarkTools:
C    - Added FLOPS (INTEGER*8) output parameter for exact flop counting
C    - Inlined helper routines (DLAE2, DLAMCH, DLANST, DLAPY2, DLASCL,
C      DLASRT) to eliminate external link dependencies
C
C> \brief \b DSTERF_FLOPS
C>
C> DSTERF_FLOPS computes all eigenvalues of a symmetric tridiagonal
C> matrix using the Pal-Walker-Kahan variant of the QL or QR algorithm,
C> and returns the total floating-point operation count in FLOPS.
C>
C> \param[in] N
C>   INTEGER. The order of the matrix. N >= 0.
C>
C> \param[in,out] D
C>   DOUBLE PRECISION array, dimension (N).
C>   On entry, the diagonal elements. On exit, eigenvalues in
C>   ascending order (if INFO = 0).
C>
C> \param[in,out] E
C>   DOUBLE PRECISION array, dimension (N-1).
C>   On entry, subdiagonal elements. On exit, destroyed.
C>
C> \param[out] INFO
C>   INTEGER. = 0: success. < 0: illegal argument. > 0: convergence
C>   failure (INFO elements of E did not converge to zero).
C>
C> \param[out] FLOPS
C>   INTEGER*8. Total floating-point operations performed.
C
      SUBROUTINE DSTERF_FLOPS( N, D, E, INFO, FLOPS )
      IMPLICIT NONE
C
C     .. Scalar Arguments ..
      INTEGER            INFO, N
      INTEGER*8          FLOPS
C     ..
C     .. Array Arguments ..
      DOUBLE PRECISION   D( * ), E( * )
C     ..
C
C     .. Parameters ..
      DOUBLE PRECISION   ZERO, ONE, TWO, THREE
      PARAMETER          ( ZERO = 0.0D0, ONE = 1.0D0, TWO = 2.0D0,
     $                   THREE = 3.0D0 )
      INTEGER            MAXIT
      PARAMETER          ( MAXIT = 30 )
C     ..
C     .. Local Scalars ..
      INTEGER            I, ISCALE, JTOT, L, L1, LEND, LENDSV, LSV, M,
     $                   NMAXIT, J, NN
      DOUBLE PRECISION   ALPHA, ANORM, BB, C, EPS, EPS2, GAMMA, OLDC,
     $                   OLDGAM, P, R, RT1, RT2, RTE, S, SAFMAX, SAFMIN,
     $                   SIGMA, SSFMAX, SSFMIN, RMAX,
     $                   AB, ACMN, ACMX, ADF, DF, TB, RT, SM,
     $                   CFROM, CTO, CTOC, CFROMC, CFROM1, CTO1, MUL,
     $                   BIGNUM, SMLNUM, TMP,
     $                   XABS, YABS, W, Z, HUGEVAL
      LOGICAL            DONE
C     ..
C     .. Intrinsic Functions ..
      INTRINSIC          ABS, SIGN, SQRT, MAX, MIN, LOG, INT
C     ..
C     .. Executable Statements ..
C
C     Test the input parameters.
C
      INFO = 0
      FLOPS = 0
C
C     Quick return if possible
C
      IF( N.LT.0 ) THEN
         INFO = -1
         RETURN
      END IF
      IF( N.LE.1 )
     $   RETURN
C
C     ---- Inlined DLAMCH constants (IEEE-754 double precision) ----
C     EPS  = DLAMCH('E') = epsilon/2 = 2^{-53} = 1.11022302462516D-16
C     SAFMIN = DLAMCH('S') = 2^{-1022} = 2.22507385850720D-308
C     RMAX = DLAMCH('O') = (1 - 2^{-53}) * 2^{1023}
C
      EPS = 1.11022302462515654D-16
      EPS2 = EPS**2
      SAFMIN = 2.22507385850720138D-308
      SAFMAX = ONE / SAFMIN
      SSFMAX = SQRT( SAFMAX ) / THREE
      SSFMIN = SQRT( SAFMIN ) / EPS2
      RMAX = 1.79769313486231571D+308
C
C     Compute the eigenvalues of the tridiagonal matrix.
C
      NMAXIT = N*MAXIT
      SIGMA = ZERO
      JTOT = 0
C
C     Determine where the matrix splits and choose QL or QR iteration
C     for each block, according to whether top or bottom diagonal
C     element is smaller.
C
      L1 = 1
C
   10 CONTINUE
      IF( L1.GT.N )
     $   GO TO 170
      IF( L1.GT.1 )
     $   E( L1-1 ) = ZERO
      DO 20 M = L1, N - 1
         IF( ABS( E( M ) ).LE.( SQRT( ABS( D( M ) ) )*SQRT( ABS( D( M+
     $       1 ) ) ) )*EPS ) THEN
            E( M ) = ZERO
            GO TO 30
         END IF
   20 CONTINUE
      M = N
C
   30 CONTINUE
      L = L1
      LSV = L
      LEND = M
      LENDSV = LEND
      L1 = M + 1
      IF( LEND.EQ.L )
     $   GO TO 10
C
C     ---- Inlined DLANST('M', ...) for max-norm ----
C
      ANORM = ZERO
      NN = LEND - L + 1
      DO 31 I = L, L + NN - 1
         TMP = ABS( D( I ) )
         IF( TMP.GT.ANORM ) ANORM = TMP
   31 CONTINUE
      DO 32 I = L, L + NN - 2
         TMP = ABS( E( I ) )
         IF( TMP.GT.ANORM ) ANORM = TMP
   32 CONTINUE
C
      ISCALE = 0
      IF( ANORM.EQ.ZERO )
     $   GO TO 10
      IF( (ANORM.GT.SSFMAX) ) THEN
         ISCALE = 1
C
C        ---- Inlined DLASCL('G',...) for D: scale by SSFMAX/ANORM ----
C
         CFROMC = ANORM
         CTOC = SSFMAX
         DONE = .FALSE.
   33    CONTINUE
         IF( .NOT.DONE ) THEN
            CFROM1 = CFROMC * SAFMIN
            IF( CFROM1.EQ.CFROMC ) THEN
               MUL = CTOC / CFROMC
               DONE = .TRUE.
            ELSE IF( ABS(CTOC).LE.SAFMIN ) THEN
               MUL = CTOC / (CFROMC * SAFMAX)
               DONE = .TRUE.
            ELSE IF( ABS(CFROM1).GT.ABS(CTOC) ) THEN
               MUL = SAFMIN
               CFROMC = CFROM1
            ELSE IF( ABS(CTOC)/SAFMAX.GT.ABS(CFROMC) ) THEN
               MUL = SAFMAX
               CFROMC = CTOC
            ELSE
               MUL = CTOC / CFROMC
               DONE = .TRUE.
            END IF
            DO 34 I = L, L + NN - 1
               D( I ) = D( I ) * MUL
   34       CONTINUE
            GO TO 33
         END IF
C
C        ---- Inlined DLASCL('G',...) for E: scale by SSFMAX/ANORM ----
C
         CFROMC = ANORM
         CTOC = SSFMAX
         DONE = .FALSE.
   35    CONTINUE
         IF( .NOT.DONE ) THEN
            CFROM1 = CFROMC * SAFMIN
            IF( CFROM1.EQ.CFROMC ) THEN
               MUL = CTOC / CFROMC
               DONE = .TRUE.
            ELSE IF( ABS(CTOC).LE.SAFMIN ) THEN
               MUL = CTOC / (CFROMC * SAFMAX)
               DONE = .TRUE.
            ELSE IF( ABS(CFROM1).GT.ABS(CTOC) ) THEN
               MUL = SAFMIN
               CFROMC = CFROM1
            ELSE IF( ABS(CTOC)/SAFMAX.GT.ABS(CFROMC) ) THEN
               MUL = SAFMAX
               CFROMC = CTOC
            ELSE
               MUL = CTOC / CFROMC
               DONE = .TRUE.
            END IF
            DO 36 I = L, L + NN - 2
               E( I ) = E( I ) * MUL
   36       CONTINUE
            GO TO 35
         END IF
C
      ELSE IF( ANORM.LT.SSFMIN ) THEN
         ISCALE = 2
C
C        ---- Inlined DLASCL('G',...) for D: scale by SSFMIN/ANORM ----
C
         CFROMC = ANORM
         CTOC = SSFMIN
         DONE = .FALSE.
   37    CONTINUE
         IF( .NOT.DONE ) THEN
            CFROM1 = CFROMC * SAFMIN
            IF( CFROM1.EQ.CFROMC ) THEN
               MUL = CTOC / CFROMC
               DONE = .TRUE.
            ELSE IF( ABS(CTOC).LE.SAFMIN ) THEN
               MUL = CTOC / (CFROMC * SAFMAX)
               DONE = .TRUE.
            ELSE IF( ABS(CFROM1).GT.ABS(CTOC) ) THEN
               MUL = SAFMIN
               CFROMC = CFROM1
            ELSE IF( ABS(CTOC)/SAFMAX.GT.ABS(CFROMC) ) THEN
               MUL = SAFMAX
               CFROMC = CTOC
            ELSE
               MUL = CTOC / CFROMC
               DONE = .TRUE.
            END IF
            DO 38 I = L, L + NN - 1
               D( I ) = D( I ) * MUL
   38       CONTINUE
            GO TO 37
         END IF
C
C        ---- Inlined DLASCL('G',...) for E: scale by SSFMIN/ANORM ----
C
         CFROMC = ANORM
         CTOC = SSFMIN
         DONE = .FALSE.
   39    CONTINUE
         IF( .NOT.DONE ) THEN
            CFROM1 = CFROMC * SAFMIN
            IF( CFROM1.EQ.CFROMC ) THEN
               MUL = CTOC / CFROMC
               DONE = .TRUE.
            ELSE IF( ABS(CTOC).LE.SAFMIN ) THEN
               MUL = CTOC / (CFROMC * SAFMAX)
               DONE = .TRUE.
            ELSE IF( ABS(CFROM1).GT.ABS(CTOC) ) THEN
               MUL = SAFMIN
               CFROMC = CFROM1
            ELSE IF( ABS(CTOC)/SAFMAX.GT.ABS(CFROMC) ) THEN
               MUL = SAFMAX
               CFROMC = CTOC
            ELSE
               MUL = CTOC / CFROMC
               DONE = .TRUE.
            END IF
            DO 41 I = L, L + NN - 2
               E( I ) = E( I ) * MUL
   41       CONTINUE
            GO TO 39
         END IF
      END IF
C
      DO 40 I = L, LEND - 1
         E( I ) = E( I )**2
   40 CONTINUE
C
C     Choose between QL and QR iteration
C
      IF( ABS( D( LEND ) ).LT.ABS( D( L ) ) ) THEN
         LEND = LSV
         L = LENDSV
      END IF
C
      IF( LEND.GE.L ) THEN
C
C        QL Iteration
C
C        Look for small subdiagonal element.
C
   50    CONTINUE
         IF( L.NE.LEND ) THEN
            DO 60 M = L, LEND - 1
               IF( ABS( E( M ) ).LE.EPS2*ABS( D( M )*D( M+1 ) ) )
     $            GO TO 70
*              D(M)*D(M+1) = 1 MUL, EPS2*result = 1 MUL (hoisted)
               FLOPS = FLOPS + 2
   60       CONTINUE
         END IF
         M = LEND
C
   70    CONTINUE
         IF( M.LT.LEND )
     $      E( M ) = ZERO
         P = D( L )
         IF( M.EQ.L )
     $      GO TO 90
C
C        If remaining matrix is 2 by 2, use inlined DLAE2.
C
         IF( M.EQ.L+1 ) THEN
            RTE = SQRT( E( L ) )
C
C           ---- Inlined DLAE2(D(L), RTE, D(L+1), RT1, RT2) ----
C
            SM  = D( L ) + D( L+1 )
            DF  = D( L ) - D( L+1 )
            ADF = ABS( DF )
            TB  = RTE + RTE
            AB  = ABS( TB )
            ACMX = D( L )
            ACMN = D( L+1 )
            IF( ABS( D( L+1 ) ).GT.ABS( D( L ) ) ) THEN
               ACMX = D( L+1 )
               ACMN = D( L )
            END IF
            IF( ADF.GT.AB ) THEN
               RT = ADF * SQRT( ONE + ( AB / ADF )**2 )
            ELSE IF( ADF.LT.AB ) THEN
               RT = AB * SQRT( ONE + ( ADF / AB )**2 )
            ELSE
               RT = AB * SQRT( TWO )
            END IF
            IF( SM.LT.ZERO ) THEN
               RT1 = 0.5D0 * ( SM - RT )
               RT2 = ( ACMX / RT1 ) * ACMN - ( RTE / RT1 ) * RTE
            ELSE IF( SM.GT.ZERO ) THEN
               RT1 = 0.5D0 * ( SM + RT )
               RT2 = ( ACMX / RT1 ) * ACMN - ( RTE / RT1 ) * RTE
            ELSE
               RT1 =  0.5D0 * RT
               RT2 = -0.5D0 * RT
            END IF
C
*           14 ops: SQRT(E(L))=SQRT, SM=ADD, DF=SUB, TB=ADD,
*           hypotenuse: DIV + FMA + SQRT + MUL = 4,
*           RT1: SUB/ADD + MUL = 2,
*           RT2: 2 DIV + MUL + FMA = 4
            FLOPS = FLOPS + 14
C
            D( L ) = RT1
            D( L+1 ) = RT2
            E( L ) = ZERO
            L = L + 2
            IF( L.LE.LEND )
     $         GO TO 50
            GO TO 150
         END IF
C
         IF( JTOT.EQ.NMAXIT )
     $      GO TO 150
         JTOT = JTOT + 1
C
C        Form shift.
C
         RTE = SQRT( E( L ) )
         SIGMA = ( D( L+1 )-P ) / ( TWO*RTE )
C
C        ---- Inlined DLAPY2(SIGMA, ONE) ----
C
         XABS = ABS( SIGMA )
         YABS = ONE
         W = MAX( XABS, YABS )
         Z = MIN( XABS, YABS )
         IF( Z.EQ.ZERO ) THEN
            R = W
         ELSE
            R = W * SQRT( ONE + ( Z / W )**2 )
         END IF
C
         SIGMA = P - ( RTE / ( SIGMA+SIGN( R, SIGMA ) ) )
C        13 ops: SQRT(E(L))=SQRT,
C        (D(L+1)-P)/(TWO*RTE) = SUB + MUL + DIV = 3,
C        DLAPY2: Z/W=DIV, (Z/W)^2+ONE=FMA, SQRT, W*result=MUL = 4,
C        P-RTE/(SIGMA+SIGN(R,SIGMA)) = ADD + DIV + SUB = 3,
C        GAMMA=D(M)-SIGMA = SUB, P=GAMMA*GAMMA = MUL = 2
         FLOPS = FLOPS + 13
C
         C = ONE
         S = ZERO
         GAMMA = D( M ) - SIGMA
         P = GAMMA*GAMMA
C
C        Inner loop
C
         DO 80 I = M - 1, L, -1
            BB = E( I )
            R = P + BB
            IF( I.NE.M-1 )
     $         E( I+1 ) = S*R
            OLDC = C
            C = P / R
            S = BB / R
            OLDGAM = GAMMA
            ALPHA = D( I )
            GAMMA = C*( ALPHA-SIGMA ) - S*OLDGAM
            D( I+1 ) = OLDGAM + ( ALPHA-GAMMA )
            IF( C.NE.ZERO ) THEN
               P = ( GAMMA*GAMMA ) / C
            ELSE
               P = OLDC*BB
            END IF
*           10 ops on first iter (I=M-1): S*R skipped by guard
*           11 ops otherwise: ADD, MUL(S*R), 2 DIV, SUB+MUL+FMA,
*             SUB+ADD, MUL+DIV
            IF( I.EQ.M-1 ) THEN
               FLOPS = FLOPS + 10
            ELSE
               FLOPS = FLOPS + 11
            END IF
   80    CONTINUE
C
         E( L ) = S*P
         D( L ) = SIGMA + GAMMA
         FLOPS = FLOPS + 2
         GO TO 50
C
C        Eigenvalue found.
C
   90    CONTINUE
         D( L ) = P
C
         L = L + 1
         IF( L.LE.LEND )
     $      GO TO 50
         GO TO 150
C
      ELSE
C
C        QR Iteration
C
C        Look for small superdiagonal element.
C
  100    CONTINUE
         DO 110 M = L, LEND + 1, -1
            IF( ABS( E( M-1 ) ).LE.EPS2*ABS( D( M )*D( M-1 ) ) )
     $         GO TO 120
*           D(M)*D(M-1) = 1 MUL, EPS2*result = 1 MUL (hoisted)
            FLOPS = FLOPS + 2
  110    CONTINUE
         M = LEND
C
  120    CONTINUE
         IF( M.GT.LEND )
     $      E( M-1 ) = ZERO
         P = D( L )
         IF( M.EQ.L )
     $      GO TO 140
C
C        If remaining matrix is 2 by 2, use inlined DLAE2.
C
         IF( M.EQ.L-1 ) THEN
            RTE = SQRT( E( L-1 ) )
C
C           ---- Inlined DLAE2(D(L), RTE, D(L-1), RT1, RT2) ----
C
            SM  = D( L ) + D( L-1 )
            DF  = D( L ) - D( L-1 )
            ADF = ABS( DF )
            TB  = RTE + RTE
            AB  = ABS( TB )
            ACMX = D( L )
            ACMN = D( L-1 )
            IF( ABS( D( L-1 ) ).GT.ABS( D( L ) ) ) THEN
               ACMX = D( L-1 )
               ACMN = D( L )
            END IF
            IF( ADF.GT.AB ) THEN
               RT = ADF * SQRT( ONE + ( AB / ADF )**2 )
            ELSE IF( ADF.LT.AB ) THEN
               RT = AB * SQRT( ONE + ( ADF / AB )**2 )
            ELSE
               RT = AB * SQRT( TWO )
            END IF
            IF( SM.LT.ZERO ) THEN
               RT1 = 0.5D0 * ( SM - RT )
               RT2 = ( ACMX / RT1 ) * ACMN - ( RTE / RT1 ) * RTE
            ELSE IF( SM.GT.ZERO ) THEN
               RT1 = 0.5D0 * ( SM + RT )
               RT2 = ( ACMX / RT1 ) * ACMN - ( RTE / RT1 ) * RTE
            ELSE
               RT1 =  0.5D0 * RT
               RT2 = -0.5D0 * RT
            END IF
C
*           14 ops: SQRT(E(L-1))=SQRT, SM=ADD, DF=SUB, TB=ADD,
*           hypotenuse: DIV + FMA + SQRT + MUL = 4,
*           RT1: SUB/ADD + MUL = 2,
*           RT2: 2 DIV + MUL + FMA = 4
            FLOPS = FLOPS + 14
C
            D( L ) = RT1
            D( L-1 ) = RT2
            E( L-1 ) = ZERO
            L = L - 2
            IF( L.GE.LEND )
     $         GO TO 100
            GO TO 150
         END IF
C
         IF( JTOT.EQ.NMAXIT )
     $      GO TO 150
         JTOT = JTOT + 1
C
C        Form shift.
C
         RTE = SQRT( E( L-1 ) )
         SIGMA = ( D( L-1 )-P ) / ( TWO*RTE )
C
C        ---- Inlined DLAPY2(SIGMA, ONE) ----
C
         XABS = ABS( SIGMA )
         YABS = ONE
         W = MAX( XABS, YABS )
         Z = MIN( XABS, YABS )
         IF( Z.EQ.ZERO ) THEN
            R = W
         ELSE
            R = W * SQRT( ONE + ( Z / W )**2 )
         END IF
C
         SIGMA = P - ( RTE / ( SIGMA+SIGN( R, SIGMA ) ) )
C        13 ops: SQRT(E(L-1))=SQRT,
C        (D(L-1)-P)/(TWO*RTE) = SUB + MUL + DIV = 3,
C        DLAPY2: Z/W=DIV, (Z/W)^2+ONE=FMA, SQRT, W*result=MUL = 4,
C        P-RTE/(SIGMA+SIGN(R,SIGMA)) = ADD + DIV + SUB = 3,
C        GAMMA=D(M)-SIGMA = SUB, P=GAMMA*GAMMA = MUL = 2
         FLOPS = FLOPS + 13
C
         C = ONE
         S = ZERO
         GAMMA = D( M ) - SIGMA
         P = GAMMA*GAMMA
C
C        Inner loop
C
         DO 130 I = M, L - 1
            BB = E( I )
            R = P + BB
            IF( I.NE.M )
     $         E( I-1 ) = S*R
            OLDC = C
            C = P / R
            S = BB / R
            OLDGAM = GAMMA
            ALPHA = D( I+1 )
            GAMMA = C*( ALPHA-SIGMA ) - S*OLDGAM
            D( I ) = OLDGAM + ( ALPHA-GAMMA )
            IF( C.NE.ZERO ) THEN
               P = ( GAMMA*GAMMA ) / C
            ELSE
               P = OLDC*BB
            END IF
*           10 ops on first iter (I=M): S*R skipped by guard
*           11 ops otherwise
            IF( I.EQ.M ) THEN
               FLOPS = FLOPS + 10
            ELSE
               FLOPS = FLOPS + 11
            END IF
  130    CONTINUE
C
         E( L-1 ) = S*P
         D( L ) = SIGMA + GAMMA
         FLOPS = FLOPS + 2
         GO TO 100
C
C        Eigenvalue found.
C
  140    CONTINUE
         D( L ) = P
C
         L = L - 1
         IF( L.GE.LEND )
     $      GO TO 100
         GO TO 150
C
      END IF
C
C     Undo scaling if necessary
C
  150 CONTINUE
      IF( ISCALE.EQ.1 ) THEN
C
C        ---- Inlined DLASCL('G',...): unscale D by ANORM/SSFMAX ----
C
         CFROMC = SSFMAX
         CTOC = ANORM
         DONE = .FALSE.
  151    CONTINUE
         IF( .NOT.DONE ) THEN
            CFROM1 = CFROMC * SAFMIN
            IF( CFROM1.EQ.CFROMC ) THEN
               MUL = CTOC / CFROMC
               DONE = .TRUE.
            ELSE IF( ABS(CTOC).LE.SAFMIN ) THEN
               MUL = CTOC / (CFROMC * SAFMAX)
               DONE = .TRUE.
            ELSE IF( ABS(CFROM1).GT.ABS(CTOC) ) THEN
               MUL = SAFMIN
               CFROMC = CFROM1
            ELSE IF( ABS(CTOC)/SAFMAX.GT.ABS(CFROMC) ) THEN
               MUL = SAFMAX
               CFROMC = CTOC
            ELSE
               MUL = CTOC / CFROMC
               DONE = .TRUE.
            END IF
            DO 152 I = LSV, LENDSV
               D( I ) = D( I ) * MUL
  152       CONTINUE
            GO TO 151
         END IF
      END IF
      IF( ISCALE.EQ.2 ) THEN
C
C        ---- Inlined DLASCL('G',...): unscale D by ANORM/SSFMIN ----
C
         CFROMC = SSFMIN
         CTOC = ANORM
         DONE = .FALSE.
  153    CONTINUE
         IF( .NOT.DONE ) THEN
            CFROM1 = CFROMC * SAFMIN
            IF( CFROM1.EQ.CFROMC ) THEN
               MUL = CTOC / CFROMC
               DONE = .TRUE.
            ELSE IF( ABS(CTOC).LE.SAFMIN ) THEN
               MUL = CTOC / (CFROMC * SAFMAX)
               DONE = .TRUE.
            ELSE IF( ABS(CFROM1).GT.ABS(CTOC) ) THEN
               MUL = SAFMIN
               CFROMC = CFROM1
            ELSE IF( ABS(CTOC)/SAFMAX.GT.ABS(CFROMC) ) THEN
               MUL = SAFMAX
               CFROMC = CTOC
            ELSE
               MUL = CTOC / CFROMC
               DONE = .TRUE.
            END IF
            DO 154 I = LSV, LENDSV
               D( I ) = D( I ) * MUL
  154       CONTINUE
            GO TO 153
         END IF
      END IF
C
C     Check for no convergence to an eigenvalue after a total
C     of N*MAXIT iterations.
C
      IF( JTOT.LT.NMAXIT )
     $   GO TO 10
      DO 160 I = 1, N - 1
         IF( E( I ).NE.ZERO )
     $      INFO = INFO + 1
  160 CONTINUE
      GO TO 180
C
C     ---- Inlined DLASRT('I', N, D, INFO) — insertion sort ----
C
  170 CONTINUE
      DO 176 I = 2, N
         TMP = D( I )
         J = I - 1
  174    CONTINUE
         IF( J.GE.1 ) THEN
            IF( D( J ).GT.TMP ) THEN
               D( J+1 ) = D( J )
               J = J - 1
               GO TO 174
            END IF
         END IF
         D( J+1 ) = TMP
  176 CONTINUE
C
  180 CONTINUE
      RETURN
C
C     End of DSTERF_FLOPS
C
      END
