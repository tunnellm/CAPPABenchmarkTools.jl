C  Modified from LAPACK's DSTEV/DSTEQR (implicit QL/QR with Givens rotations).
C  Original: Copyright (c) 1992-2024 The University of Tennessee.
C            All rights reserved.
C  Licensed under the 3-clause BSD license (see LICENSE).
C  Modifications by CAPPABenchmarkTools:
C    - Always computes eigenvectors (no JOBZ/COMPZ parameter)
C    - Added FLOPS (INTEGER*8) output parameter for exact flop counting
C    - Inlined helper routines (DLAEV2, DLARTG, DLASR, DLAMCH, DLANST,
C      DLASCL, DLASET, DLAPY2) to eliminate external link dependencies
C
C> \brief \b DSTEV_FLOPS
C>
C> DSTEV_FLOPS computes all eigenvalues and eigenvectors of a symmetric
C> tridiagonal matrix using the implicit QL or QR method with Givens
C> rotations (LAPACK's DSTEQR algorithm), and returns the total
C> floating-point operation count in FLOPS.
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
C> \param[out] Z
C>   DOUBLE PRECISION array, dimension (LDZ, N).
C>   On exit, the orthonormal eigenvectors: column i is the
C>   eigenvector for eigenvalue D(i).
C>
C> \param[in] LDZ
C>   INTEGER. Leading dimension of Z. LDZ >= max(1, N).
C>
C> \param[out] WORK
C>   DOUBLE PRECISION array, dimension (max(1, 2*N-2)).
C>   Workspace.
C>
C> \param[out] INFO
C>   INTEGER. = 0: success. < 0: illegal argument. > 0: convergence
C>   failure (INFO elements of E did not converge to zero).
C>
C> \param[out] FLOPS
C>   INTEGER*8. Total floating-point operations performed.
C
      SUBROUTINE DSTEV_FLOPS( N, D, E, Z, LDZ, WORK, INFO, FLOPS )
      IMPLICIT NONE
C
C     .. Scalar Arguments ..
      INTEGER            INFO, LDZ, N
      INTEGER*8          FLOPS
C     ..
C     .. Array Arguments ..
      DOUBLE PRECISION   D( * ), E( * ), Z( LDZ, * ), WORK( * )
C     ..
C
C     .. Parameters ..
      DOUBLE PRECISION   ZERO, ONE, TWO, THREE, HALF
      PARAMETER          ( ZERO = 0.0D0, ONE = 1.0D0, TWO = 2.0D0,
     $                   THREE = 3.0D0, HALF = 0.5D0 )
      INTEGER            MAXIT
      PARAMETER          ( MAXIT = 30 )
C     ..
C     .. Local Scalars ..
      INTEGER            I, II, ISCALE, ISCALE0, J, JTOT, K, L, L1,
     $                   LEND, LENDM1, LENDP1, LENDSV, LM1, LSV,
     $                   M, MM, MM1, NM1, NMAXIT, IMAX, NN,
     $                   SGN1, SGN2
      DOUBLE PRECISION   ANORM, B, C, EPS, EPS2, F, G, P, R, RT1, RT2,
     $                   S, SAFMAX, SAFMIN, SSFMAX, SSFMIN, TST,
     $                   SIGMA0, TNRM, RMIN, BIGNUM, SMLNUM,
     $                   TEMP, CTEMP, STEMP,
     $                   CFROMC, CTOC, CFROM1, MUL, TMP,
     $                   XABS, YABS, W, ZZ,
     $                   SM, DF, ADF, TB, AB, ACMX, ACMN,
     $                   RT, ACS, CT, TN, CS1, SN1,
     $                   F1, G1, RTMIN, RTMAX, DD, UU, FS, GS
      LOGICAL            DONE
C     ..
C     .. Intrinsic Functions ..
      INTRINSIC          ABS, MAX, MIN, SIGN, SQRT
C     ..
C     .. Executable Statements ..
C
C     Test the input parameters.
C
      INFO = 0
      FLOPS = 0
C
      IF( N.LT.0 ) THEN
         INFO = -1
         RETURN
      END IF
      IF( LDZ.LT.MAX( 1, N ) ) THEN
         INFO = -5
         RETURN
      END IF
C
C     Quick return if possible
C
      IF( N.EQ.0 )
     $   RETURN
C
      IF( N.EQ.1 ) THEN
         Z( 1, 1 ) = ONE
         RETURN
      END IF
C
C     ---- Inlined DLAMCH constants (IEEE-754 double precision) ----
C     EPS  = DLAMCH('E') = epsilon/2 = 2^{-53}
C     SAFMIN = DLAMCH('S') = 2^{-1022}
C
      EPS = 1.11022302462515654D-16
      EPS2 = EPS**2
      SAFMIN = 2.22507385850720138D-308
      SAFMAX = ONE / SAFMIN
      SSFMAX = SQRT( SAFMAX ) / THREE
      SSFMIN = SQRT( SAFMIN ) / EPS2
      SMLNUM = SAFMIN / EPS
      BIGNUM = ONE / SMLNUM
      RMIN = SQRT( SMLNUM )
C     DLARTG constants
      RTMIN = SQRT( SAFMIN )
      RTMAX = SQRT( SAFMAX / TWO )
C
C     ======================================================================
C     DSTEV outer scaling: scale to safe range if needed
C     ======================================================================
C
C     ---- Inlined DLANST('M', N, D, E) for max norm ----
C
      TNRM = ZERO
      DO I = 1, N
         TMP = ABS( D( I ) )
         IF( TMP.GT.TNRM ) TNRM = TMP
      END DO
      DO I = 1, N - 1
         TMP = ABS( E( I ) )
         IF( TMP.GT.TNRM ) TNRM = TMP
      END DO
C
      ISCALE0 = 0
      IF( TNRM.GT.ZERO .AND. TNRM.LT.RMIN ) THEN
         ISCALE0 = 1
         SIGMA0 = RMIN / TNRM
      ELSE IF( TNRM.GT.SQRT( BIGNUM ) ) THEN
         ISCALE0 = 1
         SIGMA0 = SQRT( BIGNUM ) / TNRM
      END IF
      IF( ISCALE0.EQ.1 ) THEN
         DO I = 1, N
            D( I ) = D( I ) * SIGMA0
         END DO
         DO I = 1, N - 1
            E( I ) = E( I ) * SIGMA0
         END DO
      END IF
C
C     ======================================================================
C     Initialize Z to identity (inlined DLASET('Full', N, N, 0, 1, Z, LDZ))
C     ======================================================================
C
      DO J = 1, N
         DO I = 1, N
            Z( I, J ) = ZERO
         END DO
         Z( J, J ) = ONE
      END DO
C
C     ======================================================================
C     Main DSTEQR iteration (always with eigenvectors, ICOMPZ=2)
C     ======================================================================
C
      NMAXIT = N * MAXIT
      JTOT = 0
      NM1 = N - 1
C
      L1 = 1
C
   10 CONTINUE
      IF( L1.GT.N )
     $   GO TO 160
      IF( L1.GT.1 )
     $   E( L1-1 ) = ZERO
      IF( L1.LE.NM1 ) THEN
         DO 20 M = L1, NM1
            TST = ABS( E( M ) )
            IF( TST.EQ.ZERO )
     $         GO TO 30
            IF( TST.LE.( SQRT( ABS( D( M ) ) )*
     $          SQRT( ABS( D( M+1 ) ) ) )*EPS ) THEN
               E( M ) = ZERO
               GO TO 30
            END IF
   20    CONTINUE
      END IF
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
C     ---- Inlined DLANST('M', ...) for max-norm of subblock ----
C
      ANORM = ZERO
      NN = LEND - L + 1
      DO I = L, L + NN - 1
         TMP = ABS( D( I ) )
         IF( TMP.GT.ANORM ) ANORM = TMP
      END DO
      DO I = L, L + NN - 2
         TMP = ABS( E( I ) )
         IF( TMP.GT.ANORM ) ANORM = TMP
      END DO
C
      ISCALE = 0
      IF( ANORM.EQ.ZERO )
     $   GO TO 10
      IF( ANORM.GT.SSFMAX ) THEN
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
            DO I = L, L + NN - 1
               D( I ) = D( I ) * MUL
            END DO
            GO TO 33
         END IF
C
C        ---- Inlined DLASCL('G',...) for E ----
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
            DO I = L, L + NN - 2
               E( I ) = E( I ) * MUL
            END DO
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
            DO I = L, L + NN - 1
               D( I ) = D( I ) * MUL
            END DO
            GO TO 37
         END IF
C
C        ---- Inlined DLASCL('G',...) for E ----
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
            DO I = L, L + NN - 2
               E( I ) = E( I ) * MUL
            END DO
            GO TO 39
         END IF
      END IF
C
C     Choose between QL and QR iteration
C
      IF( ABS( D( LEND ) ).LT.ABS( D( L ) ) ) THEN
         LEND = LSV
         L = LENDSV
      END IF
C
      IF( LEND.GT.L ) THEN
C
C        ================================================================
C        QL Iteration
C        ================================================================
C
C        Look for small subdiagonal element.
C
   40    CONTINUE
         IF( L.NE.LEND ) THEN
            LENDM1 = LEND - 1
            DO 50 M = L, LENDM1
               TST = ABS( E( M ) )**2
               IF( TST.LE.( EPS2*ABS( D( M ) ) )*ABS( D( M+1 ) )+
     $             SAFMIN ) GO TO 60
*              |E(M)|^2=MUL, EPS2*|D(M)|=MUL (hoisted),
*              result*|D(M+1)|+SAFMIN = FMA => 3
               FLOPS = FLOPS + 3
   50       CONTINUE
         END IF
C
         M = LEND
C
   60    CONTINUE
         IF( M.LT.LEND )
     $      E( M ) = ZERO
         P = D( L )
         IF( M.EQ.L )
     $      GO TO 80
C
C        If remaining matrix is 2-by-2, use inlined DLAEV2
C        to compute eigenvalues and eigenvector rotation.
C
         IF( M.EQ.L+1 ) THEN
C
C           ---- Inlined DLAEV2(D(L), E(L), D(L+1), RT1, RT2, CS1, SN1) ----
C
            SM  = D( L ) + D( L+1 )
            DF  = D( L ) - D( L+1 )
            ADF = ABS( DF )
            TB  = E( L ) + E( L )
            AB  = ABS( TB )
            IF( ABS( D( L ) ).GT.ABS( D( L+1 ) ) ) THEN
               ACMX = D( L )
               ACMN = D( L+1 )
            ELSE
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
               RT1 = HALF * ( SM - RT )
               SGN1 = -1
               RT2 = ( ACMX / RT1 )*ACMN -
     $               ( E( L ) / RT1 )*E( L )
            ELSE IF( SM.GT.ZERO ) THEN
               RT1 = HALF * ( SM + RT )
               SGN1 = 1
               RT2 = ( ACMX / RT1 )*ACMN -
     $               ( E( L ) / RT1 )*E( L )
            ELSE
               RT1 = HALF * RT
               RT2 = -HALF * RT
               SGN1 = 1
            END IF
C           Compute eigenvector rotation
            IF( DF.GE.ZERO ) THEN
               CS1 = DF + RT
               SGN2 = 1
            ELSE
               CS1 = DF - RT
               SGN2 = -1
            END IF
            ACS = ABS( CS1 )
            IF( ACS.GT.AB ) THEN
               CT = -TB / CS1
               SN1 = ONE / SQRT( ONE + CT*CT )
               CS1 = CT * SN1
            ELSE
               IF( AB.EQ.ZERO ) THEN
                  CS1 = ONE
                  SN1 = ZERO
               ELSE
                  TN = -CS1 / TB
                  CS1 = ONE / SQRT( ONE + TN*TN )
                  SN1 = TN * CS1
               END IF
            END IF
            IF( SGN1.EQ.SGN2 ) THEN
               TN = CS1
               CS1 = -SN1
               SN1 = TN
            END IF
C
*           19 ops: SM+DF+TB=3, hypotenuse: DIV+FMA+SQRT+MUL=4,
*           RT1: ADD/SUB+MUL=2, RT2: 2DIV+MUL+FMA=4,
*           eigvec: ADD/SUB + DIV+FMA+SQRT+DIV+MUL = 6
            FLOPS = FLOPS + 19
C
            D( L ) = RT1
            D( L+1 ) = RT2
            E( L ) = ZERO
C
C           ---- Inlined DLASR('R','V','B', N, 2, ...) ----
C           Apply single rotation to Z(:,L) and Z(:,L+1)
C
            DO I = 1, N
               TEMP = Z( I, L+1 )
               Z( I, L+1 ) = CS1*TEMP - SN1*Z( I, L )
               Z( I, L ) = SN1*TEMP + CS1*Z( I, L )
            END DO
*           4*N: per row c*x-s*y = MUL+FMA, s*x+c*y = MUL+FMA
            FLOPS = FLOPS + 4*N
C
            L = L + 2
            IF( L.LE.LEND )
     $         GO TO 40
            GO TO 140
         END IF
C
         IF( JTOT.EQ.NMAXIT )
     $      GO TO 140
         JTOT = JTOT + 1
C
C        Form shift.
C
         G = ( D( L+1 ) - P ) / ( TWO*E( L ) )
C
C        ---- Inlined DLAPY2(G, ONE) ----
C
         XABS = ABS( G )
         YABS = ONE
         W = MAX( XABS, YABS )
         ZZ = MIN( XABS, YABS )
         IF( ZZ.EQ.ZERO ) THEN
            R = W
         ELSE
            R = W * SQRT( ONE + ( ZZ / W )**2 )
         END IF
C
         G = D( M ) - P + ( E( L ) / ( G + SIGN( R, G ) ) )
C        11 ops: (D(L+1)-P)/(TWO*E(L)) = SUB+MUL+DIV = 3,
C        DLAPY2: Z/W=DIV, (Z/W)^2+ONE=FMA, SQRT, W*result=MUL = 4,
C        D(M)-P+E(L)/(G+SIGN(R,G)) = ADD+DIV+SUB+ADD = 4
         FLOPS = FLOPS + 11
C
         S = ONE
         C = ONE
         P = ZERO
C
C        Inner loop
C
         MM1 = M - 1
         DO 70 I = MM1, L, -1
            F = S * E( I )
            B = C * E( I )
C
C           ---- Inlined DLARTG(G, F, C, S, R) ----
C           DLARTG's first arg = G, second arg = F
C
            F1 = ABS( G )
            G1 = ABS( F )
            IF( F.EQ.ZERO ) THEN
               C = ONE
               S = ZERO
               R = G
            ELSE IF( G.EQ.ZERO ) THEN
               C = ZERO
               S = SIGN( ONE, F )
               R = G1
            ELSE IF( F1.GT.RTMIN .AND. F1.LT.RTMAX .AND.
     $               G1.GT.RTMIN .AND. G1.LT.RTMAX ) THEN
               DD = SQRT( G*G + F*F )
               C = F1 / DD
               R = SIGN( DD, G )
               S = F / R
            ELSE
               UU = MIN( SAFMAX, MAX( SAFMIN, F1, G1 ) )
               FS = G / UU
               GS = F / UU
               DD = SQRT( FS*FS + GS*GS )
               C = ABS( FS ) / DD
               R = SIGN( DD, G )
               S = GS / R
               R = R * UU
            END IF
C
            IF( I.NE.M-1 )
     $         E( I+1 ) = R
            G = D( I+1 ) - P
            R = ( D( I ) - G )*S + TWO*C*B
            P = S * R
            D( I+1 ) = G + P
            G = C*R - B
C
C           Save rotations for later application to Z
C
            WORK( I ) = C
            WORK( N-1+I ) = -S
C
*           15 ops: F=S*E(I)=MUL, B=C*E(I)=MUL,
*           DLARTG normal: G*G=MUL, F*F+G*G=FMA, SQRT, F1/DD=DIV,
*             F/R=DIV = 5,
*           G=D(I+1)-P=SUB,
*           R=(D(I)-G)*S+TWO*C*B = SUB+MUL+MUL+FMA = 4,
*           P=S*R=MUL, D(I+1)=G+P=ADD, G=C*R-B=FMA
            FLOPS = FLOPS + 15
   70    CONTINUE
C
C        ---- Inlined DLASR('R','V','B', N, MM, ...) ----
C        Apply saved rotations backward to Z(:, L:M)
C
         MM = M - L + 1
         DO J = MM - 1, 1, -1
            CTEMP = WORK( L+J-1 )
            STEMP = WORK( N-1+L+J-1 )
            IF( ( CTEMP.NE.ONE ) .OR. ( STEMP.NE.ZERO ) ) THEN
               DO I = 1, N
                  TEMP = Z( I, L+J )
                  Z( I, L+J ) = CTEMP*TEMP - STEMP*Z( I, L+J-1 )
                  Z( I, L+J-1 ) = STEMP*TEMP + CTEMP*Z( I, L+J-1 )
               END DO
*              4*N: per row c*x-s*y = MUL+FMA, s*x+c*y = MUL+FMA
               FLOPS = FLOPS + 4*N
            END IF
         END DO
C
         D( L ) = D( L ) - P
         E( L ) = G
         FLOPS = FLOPS + 1
         GO TO 40
C
C        Eigenvalue found.
C
   80    CONTINUE
         D( L ) = P
C
         L = L + 1
         IF( L.LE.LEND )
     $      GO TO 40
         GO TO 140
C
      ELSE
C
C        ================================================================
C        QR Iteration
C        ================================================================
C
C        Look for small superdiagonal element.
C
   90    CONTINUE
         IF( L.NE.LEND ) THEN
            LENDP1 = LEND + 1
            DO 100 M = L, LENDP1, -1
               TST = ABS( E( M-1 ) )**2
               IF( TST.LE.( EPS2*ABS( D( M ) ) )*ABS( D( M-1 ) )+
     $             SAFMIN ) GO TO 110
*              |E(M-1)|^2=MUL, EPS2*|D(M)|=MUL (hoisted),
*              result*|D(M-1)|+SAFMIN = FMA => 3
               FLOPS = FLOPS + 3
  100       CONTINUE
         END IF
C
         M = LEND
C
  110    CONTINUE
         IF( M.GT.LEND )
     $      E( M-1 ) = ZERO
         P = D( L )
         IF( M.EQ.L )
     $      GO TO 130
C
C        If remaining matrix is 2-by-2, use inlined DLAEV2.
C
         IF( M.EQ.L-1 ) THEN
C
C           ---- Inlined DLAEV2(D(L-1), E(L-1), D(L), RT1, RT2, CS1, SN1) ----
C
            SM  = D( L-1 ) + D( L )
            DF  = D( L-1 ) - D( L )
            ADF = ABS( DF )
            TB  = E( L-1 ) + E( L-1 )
            AB  = ABS( TB )
            IF( ABS( D( L-1 ) ).GT.ABS( D( L ) ) ) THEN
               ACMX = D( L-1 )
               ACMN = D( L )
            ELSE
               ACMX = D( L )
               ACMN = D( L-1 )
            END IF
            IF( ADF.GT.AB ) THEN
               RT = ADF * SQRT( ONE + ( AB / ADF )**2 )
            ELSE IF( ADF.LT.AB ) THEN
               RT = AB * SQRT( ONE + ( ADF / AB )**2 )
            ELSE
               RT = AB * SQRT( TWO )
            END IF
            IF( SM.LT.ZERO ) THEN
               RT1 = HALF * ( SM - RT )
               SGN1 = -1
               RT2 = ( ACMX / RT1 )*ACMN -
     $               ( E( L-1 ) / RT1 )*E( L-1 )
            ELSE IF( SM.GT.ZERO ) THEN
               RT1 = HALF * ( SM + RT )
               SGN1 = 1
               RT2 = ( ACMX / RT1 )*ACMN -
     $               ( E( L-1 ) / RT1 )*E( L-1 )
            ELSE
               RT1 = HALF * RT
               RT2 = -HALF * RT
               SGN1 = 1
            END IF
C           Compute eigenvector rotation
            IF( DF.GE.ZERO ) THEN
               CS1 = DF + RT
               SGN2 = 1
            ELSE
               CS1 = DF - RT
               SGN2 = -1
            END IF
            ACS = ABS( CS1 )
            IF( ACS.GT.AB ) THEN
               CT = -TB / CS1
               SN1 = ONE / SQRT( ONE + CT*CT )
               CS1 = CT * SN1
            ELSE
               IF( AB.EQ.ZERO ) THEN
                  CS1 = ONE
                  SN1 = ZERO
               ELSE
                  TN = -CS1 / TB
                  CS1 = ONE / SQRT( ONE + TN*TN )
                  SN1 = TN * CS1
               END IF
            END IF
            IF( SGN1.EQ.SGN2 ) THEN
               TN = CS1
               CS1 = -SN1
               SN1 = TN
            END IF
C
*           19 ops: SM+DF+TB=3, hypotenuse: DIV+FMA+SQRT+MUL=4,
*           RT1: ADD/SUB+MUL=2, RT2: 2DIV+MUL+FMA=4,
*           eigvec: ADD/SUB + DIV+FMA+SQRT+DIV+MUL = 6
            FLOPS = FLOPS + 19
C
            D( L-1 ) = RT1
            D( L ) = RT2
            E( L-1 ) = ZERO
C
C           ---- Inlined DLASR('R','V','F', N, 2, ...) ----
C           Apply single rotation to Z(:,L-1) and Z(:,L)
C
            DO I = 1, N
               TEMP = Z( I, L )
               Z( I, L ) = CS1*TEMP - SN1*Z( I, L-1 )
               Z( I, L-1 ) = SN1*TEMP + CS1*Z( I, L-1 )
            END DO
*           4*N: per row c*x-s*y = MUL+FMA, s*x+c*y = MUL+FMA
            FLOPS = FLOPS + 4*N
C
            L = L - 2
            IF( L.GE.LEND )
     $         GO TO 90
            GO TO 140
         END IF
C
         IF( JTOT.EQ.NMAXIT )
     $      GO TO 140
         JTOT = JTOT + 1
C
C        Form shift.
C
         G = ( D( L-1 ) - P ) / ( TWO*E( L-1 ) )
C
C        ---- Inlined DLAPY2(G, ONE) ----
C
         XABS = ABS( G )
         YABS = ONE
         W = MAX( XABS, YABS )
         ZZ = MIN( XABS, YABS )
         IF( ZZ.EQ.ZERO ) THEN
            R = W
         ELSE
            R = W * SQRT( ONE + ( ZZ / W )**2 )
         END IF
C
         G = D( M ) - P + ( E( L-1 ) / ( G + SIGN( R, G ) ) )
C        11 ops: (D(L-1)-P)/(TWO*E(L-1)) = SUB+MUL+DIV = 3,
C        DLAPY2: Z/W=DIV, (Z/W)^2+ONE=FMA, SQRT, W*result=MUL = 4,
C        D(M)-P+E(L-1)/(G+SIGN(R,G)) = ADD+DIV+SUB+ADD = 4
         FLOPS = FLOPS + 11
C
         S = ONE
         C = ONE
         P = ZERO
C
C        Inner loop
C
         LM1 = L - 1
         DO 120 I = M, LM1
            F = S * E( I )
            B = C * E( I )
C
C           ---- Inlined DLARTG(G, F, C, S, R) ----
C           DLARTG's first arg = G, second arg = F
C
            F1 = ABS( G )
            G1 = ABS( F )
            IF( F.EQ.ZERO ) THEN
               C = ONE
               S = ZERO
               R = G
            ELSE IF( G.EQ.ZERO ) THEN
               C = ZERO
               S = SIGN( ONE, F )
               R = G1
            ELSE IF( F1.GT.RTMIN .AND. F1.LT.RTMAX .AND.
     $               G1.GT.RTMIN .AND. G1.LT.RTMAX ) THEN
               DD = SQRT( G*G + F*F )
               C = F1 / DD
               R = SIGN( DD, G )
               S = F / R
            ELSE
               UU = MIN( SAFMAX, MAX( SAFMIN, F1, G1 ) )
               FS = G / UU
               GS = F / UU
               DD = SQRT( FS*FS + GS*GS )
               C = ABS( FS ) / DD
               R = SIGN( DD, G )
               S = GS / R
               R = R * UU
            END IF
C
            IF( I.NE.M )
     $         E( I-1 ) = R
            G = D( I ) - P
            R = ( D( I+1 ) - G )*S + TWO*C*B
            P = S * R
            D( I ) = G + P
            G = C*R - B
C
C           Save rotations for later application to Z
C
            WORK( I ) = C
            WORK( N-1+I ) = S
C
*           15 ops: F=S*E(I)=MUL, B=C*E(I)=MUL,
*           DLARTG normal: G*G=MUL, F*F+G*G=FMA, SQRT, F1/DD=DIV,
*             F/R=DIV = 5,
*           G=D(I)-P=SUB,
*           R=(D(I+1)-G)*S+TWO*C*B = SUB+MUL+MUL+FMA = 4,
*           P=S*R=MUL, D(I)=G+P=ADD, G=C*R-B=FMA
            FLOPS = FLOPS + 15
  120    CONTINUE
C
C        ---- Inlined DLASR('R','V','F', N, MM, ...) ----
C        Apply saved rotations forward to Z(:, M:L)
C
         MM = L - M + 1
         DO J = 1, MM - 1
            CTEMP = WORK( M+J-1 )
            STEMP = WORK( N-1+M+J-1 )
            IF( ( CTEMP.NE.ONE ) .OR. ( STEMP.NE.ZERO ) ) THEN
               DO I = 1, N
                  TEMP = Z( I, M+J )
                  Z( I, M+J ) = CTEMP*TEMP - STEMP*Z( I, M+J-1 )
                  Z( I, M+J-1 ) = STEMP*TEMP + CTEMP*Z( I, M+J-1 )
               END DO
*              4*N: per row c*x-s*y = MUL+FMA, s*x+c*y = MUL+FMA
               FLOPS = FLOPS + 4*N
            END IF
         END DO
C
         D( L ) = D( L ) - P
         E( LM1 ) = G
         FLOPS = FLOPS + 1
         GO TO 90
C
C        Eigenvalue found.
C
  130    CONTINUE
         D( L ) = P
C
         L = L - 1
         IF( L.GE.LEND )
     $      GO TO 90
         GO TO 140
C
      END IF
C
C     Undo scaling if necessary
C
  140 CONTINUE
      IF( ISCALE.EQ.1 ) THEN
C
C        ---- Inlined DLASCL: unscale D by ANORM/SSFMAX ----
C
         CFROMC = SSFMAX
         CTOC = ANORM
         DONE = .FALSE.
  141    CONTINUE
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
            DO I = LSV, LENDSV
               D( I ) = D( I ) * MUL
            END DO
            GO TO 141
         END IF
C
C        ---- Inlined DLASCL: unscale E ----
C
         CFROMC = SSFMAX
         CTOC = ANORM
         DONE = .FALSE.
  142    CONTINUE
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
            DO I = LSV, LENDSV - 1
               E( I ) = E( I ) * MUL
            END DO
            GO TO 142
         END IF
      ELSE IF( ISCALE.EQ.2 ) THEN
C
C        ---- Inlined DLASCL: unscale D by ANORM/SSFMIN ----
C
         CFROMC = SSFMIN
         CTOC = ANORM
         DONE = .FALSE.
  143    CONTINUE
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
            DO I = LSV, LENDSV
               D( I ) = D( I ) * MUL
            END DO
            GO TO 143
         END IF
C
C        ---- Inlined DLASCL: unscale E ----
C
         CFROMC = SSFMIN
         CTOC = ANORM
         DONE = .FALSE.
  144    CONTINUE
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
            DO I = LSV, LENDSV - 1
               E( I ) = E( I ) * MUL
            END DO
            GO TO 144
         END IF
      END IF
C
C     Check for no convergence to an eigenvalue after a total
C     of N*MAXIT iterations.
C
      IF( JTOT.LT.NMAXIT )
     $   GO TO 10
      DO 150 I = 1, N - 1
         IF( E( I ).NE.ZERO )
     $      INFO = INFO + 1
  150 CONTINUE
      GO TO 190
C
C     ---- Selection sort + column swap (inlined DLASRT/DSWAP) ----
C     Sort eigenvalues in ascending order, swapping eigenvector columns.
C
  160 CONTINUE
      DO 180 II = 2, N
         I = II - 1
         K = I
         P = D( I )
         DO 170 J = II, N
            IF( D( J ).LT.P ) THEN
               K = J
               P = D( J )
            END IF
  170    CONTINUE
         IF( K.NE.I ) THEN
            D( K ) = D( I )
            D( I ) = P
C           ---- Inlined DSWAP(N, Z(1,I), 1, Z(1,K), 1) ----
            DO J = 1, N
               TEMP = Z( J, I )
               Z( J, I ) = Z( J, K )
               Z( J, K ) = TEMP
            END DO
         END IF
  180 CONTINUE
C
C     ---- DSTEV outer unscaling ----
C
  190 CONTINUE
      IF( ISCALE0.EQ.1 ) THEN
         IF( INFO.EQ.0 ) THEN
            IMAX = N
         ELSE
            IMAX = INFO - 1
         END IF
         DO I = 1, IMAX
            D( I ) = D( I ) / SIGMA0
         END DO
      END IF
C
      RETURN
C
C     End of DSTEV_FLOPS
C
      END
