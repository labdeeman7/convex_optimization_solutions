function [x, Qx, mse, xiPsi] = ista(b, F, G, lambda, x0, stop, tol, maxit, xt, iPsi)
% ISTA Iterative shrinkage-thresholding algorithm
%  [x, Qx, mse, xiPsi] = ista(b, F, G, lambda, x0, stop, tol, maxit, xt, iPsi)
%  Solves
%   argmin Q(x) = F(x,b) + lambda*G(x)
%  with Iterative shrinkage-thresholding algorithm (Beck, Teboulle '09)
%
%  INPUTS:
%   b      - data vector
%   F.f    - is strictly convex. ISTA uses quadratic majorization   
%            f(x) <= f(y) + df(y)^T (x-y) + L/2||x-y||^2, 
%            where L is Lipschitz constant for the gradient ie 
%            ||df(x) - df(y)|| <= L ||x-y|| forall x,y
%   G.f    - is convex hence subdifferentiable, Phi.prox is proximal operator with respect to G.f 
%   lambda - regularization parameter
%   x0     - initial solution (usually 0)
%   stop   - stopping condition, choose from
%             0 - relative variation in number of nnz components falls below tol
%             1 - relative variation in regularized objective function falls below tol
%             2 - relative variation in estimate falls below tol (default)
%             3 - value of the regularized objective function  falls below tol             
%   tol    - stopping tolerance for criterium specified with stop
%   maxit  - maximal number of ISTA iterations
%   xt     - true solution, if available the error will be computed
%   iPsi   - inverse transform if x is in basis Psi, iPsi(x) is in Cartesian basis
%  OUTPUTS:
%   x      - solution x
%   Qx     - values of Q(x_j) over the iteration
%   mse    - mean square error if true solution xt specified, otherwise ||iPsi(x)||^2/length(xt)
%   xiPsi  - iPsi(x) if iPsi specified (Cartesian basis)
%
% Copyright (C) 2013 Marta M. Betcke

% Default values of parameters
DL = 1;        % Lipschitz constant
DTOL = 1e-4;   % Tolerance for the objective function change, STOP if below
DMAXIT = 200;  % Maximal number of iterations 

% If no inverse transform given assume identity
if nargin < 10
  iPsi = @(x) x; 
end
% If solution not known, return ||x||_2 in err
if nargin < 9 
  xt = zeros(n,1);
end
% Assign default values of the parameters
if nargin < 8 
  maxit = DMAXIT;
end
if nargin < 7 
  tol = DTOL;
end
if nargin < 6
  stop = DSTOP;
end
if ~isfield(F,'L')
  F.L = DL;
end

% Function handles
Q = @(x,lambda) F.f(x,b) + lambda*G.f(x);

% Initialization
mse = zeros(1,maxit);
Qx = zeros(1,maxit);
x = x0;

% Evaluate objective function
Qx(1) = Q(x,lambda);
% Solution error
mse(1) = sum((iPsi(x)-xt).^2);

% Maximal step size 1/L, L = lambda_max(A^T A) Lipschitz constant of the gradient
t = 1/F.L;

for j = 2:maxit
  % Gradient step for F
  xnew = x - t*F.df(x,b);
  % Proximal step wrt G
  xnew = G.prox(xnew,t*lambda);
 
  % Evaluate objective function
  Qx(j) = Q(xnew,lambda);
  % Solution error   
  mse(j) = sum((iPsi(xnew)-xt).^2);

  % Stopping criterium
  switch stop
    case 0,
      % Relative variation of the number of nnz components
      criterion = sum(nnz(xnew) ~= nnz(x)) / nnz(x);
    case 1,
      % Relative variation of the regularized objective function
      criterion = abs(Qx(j-1) - Qx(j))/Qx(j-1);
    case 2,
      % Relative variation of the estimate
      criterion = norm(xnew - x)/norm(x);
    case 3,
      % Value of the objective function
      criterion = Qx(j);
    otherwise,
      error(['Invalid stopping criterion']);
  end

  % Reassign
  x = xnew;

  if criterion < tol
    break
  end    

%  %Stopping criterium
%  if abs(Qx(j-1) - Qx(j)) < tol
%    break
%  end
end

% Remove tailing 0s
mse = mse(1:j);
Qx = Qx(1:j);

% MSE
mse = mse/length(xt);

% Return solution in Cartesian basis if iPsi given
xiPsi = iPsi(x);