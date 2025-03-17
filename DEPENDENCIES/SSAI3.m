function M = SSAI3(A,n,lfil)

%        M = SSAI3(A,n,lfil);
% constructs a SPAI-type preconditioner from SPD A.
% Typically, lfil = ceil(nnz(A)/n);  = average nnz in a column of A.
% M is symmetric but might not be SPD.

% Shaked Regev and Michael Saunders
% ICME, Stanford University.

% 15 Aug 2019: Shaked's code entire_r_sparse_inverse.m from this date,
%              called from Shaked's SSAI.m.
%              A is assumed to have unit diagonals.
% 21 Apr 2020: Reformatted, renamed.  Trying it with Minres.
% 30 Apr 2020: Included in http://stanford.edu/group/SOL/software/minres/minres20.zip.
% 07 May 2020: tol added to exit loop if r is essentially 0.
%              Assumes A is scaled so norm(A) = O(1).
% 07 May 2020: Output M is exactly symmetric.
% 07 May 2020: Name changed from MSSAI to SSAI.
% 18 May 2020: SSAI2: Hardwire first time through loop k.
% 19 May 2020: Set delta = full(delta).
% 20 May 2020: SSAI3: delta = A(:,i)'*r / Anorms(i); is LS soln for 1 variable.
% 21 May 2020: Change the diag or sign of some m.

  M     = spalloc(n,n,lfil*n);
  m0    = spalloc(n,1,lfil*2);   % Allow more than lfil just in case
  itmax = 2*lfil;
  tol   = 1e-3;
  nneg  = 0;
  nzero = 0;

  Anorms = zeros(n,1);
  for j=1:n, Anorms(j) = norm(A(:,j))^2; end   % ||A(:,j)||^2

  for j=1:n
     m    = m0;         % m = 0;  r = sparse(j,1,1,n,1);      (r = ej)
   % m(j) = 1;          % delta = 1;   m(j) = m(j) + delta;   (m = ej)
   % r    = m - A(:,j); % r = r - delta*aj;
     r = sparse(j,1,1,n,1);
     i1  = 1;
     ri1 = r(1);

     for k=1:itmax
       [~,i] = max(abs(r));
       r(i1) = ri1;
       % delta = r(i)/A(i,i);   % General
       % delta = full(delta);   % Assume A(i,i)=1
       % if delta<=tol, break, end
       delta = A(:,i)'*r / Anorms(i);
       delta = full(delta);
       m(i)  = m(i) + delta;
       if nnz(m)>=lfil, break; end
       r     = r - delta*A(:,i);
       i1    = i;
       ri1   = r(i);
       r(i)  = 0;   % Don't select i next time around
     end

     Mjj    = m(j);
     if Mjj< 0, nneg  = nneg  + 1;  m    = -m; end
     if Mjj==0, nzero = nzero + 1;  m(j) =  1; end
     M(:,j) = m;
  end

  d = diag(M);           % sparse column vector
  M = (M+M')*0.5;
  M = tril(M,-1);
  M = M + M' + diag(d);  % M is exactly symmetric
  fprintf('\n Negative diags: %8i', nneg )
  fprintf('\n Zero     diags: %8i', nzero)
  fprintf('\n M symmetrized.  nnz(M) = %d\n', nnz(M))

end % function SSAI
