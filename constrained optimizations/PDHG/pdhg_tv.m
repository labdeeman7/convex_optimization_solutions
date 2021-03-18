% Perform TV denoising using adaptive PDHG.  The problem solved is
%            min_u    TV(u) + mu/2 || u - noisy ||^2
%  Inputs...
%    noisy : An 2D array containing the pixels of the noisy image
%    mu    : A scalar, the regularization parameter for the denoising
%    opts  : An optional argument for customizing the behavior of 
%              pdhg_adaptive
%  Outputs...
%    denoised : The denoised image
%    out      : A struct containing convergence data generated by the
%                function 'pdhg_adaptive' when it solved the problem.
% 

%  This File requires that 'pdhg_adaptive.m' be in your current path. 
%
%  For an explanation of PDHG and how it applies to this problem, see
%    "Adaptive Primal-Dual Hybrid Gradient Methods for Saddle-Point
%    Problems"  available at <http://arxiv.org/abs/1305.0546>
%

function [denoised, out] = pdhg_tv(noisy, mu, opts) 
   %% Setup the ingredients for PDHG
    A = @(x) grad(x); % The 2D gradient using forward differences
    At = @(y) -div(y); % The adjoint of A, the negative divergence using backward differences
    fProx = @(x,tau)(mu*noisy+x/tau)/(mu+1/tau);  % f = mu/2|| x - noisy ||^2
    gProx = @(y,sigma) projectInf(y); % g is the characteristic function of the infinity norm unit ball
   %% The initial iterates
    [rows,cols] = size(noisy);
    x0 = zeros(rows,cols);
    y0 = zeros(rows,cols,2);
   
   %%  Some optional convergence parameters
    if ~exist('opts','var')
        opts = [];
    end
    
    %%  Some optional things:  uncomment to use
         %  f1 computes the objective value of iterates
    %  opts.f1 = @(x,y,x0,y0,tau,sigma) sum(sum(sqrt(Dx(x).^2+Dy(x).^2)))+mu/2*sum(sum((x-noisy).^2));
   
    
    %% Call the solver
   [denoised ,out]= pdhg_adaptive(x0,y0,A,At,fProx,gProx,opts); 
   
   
return

%  The 2D gradient using forward differences
function g = grad(u)
 [rows cols] = size(u);
 g = zeros(rows,cols,2);
 g(:,:,1) = Dx(u);
 g(:,:,2) = Dy(u);
return

%  The 2D divergence using backwards differences.  This is the adjoint of
%  the gradient operator represented by grad().
function d = div(u)
 d = -(Dxt(u(:,:,1))+Dyt(u(:,:,2)));
return

%  Project dual variable onto the infinity-norm unit ball
function z = projectInf( z )
x = z(:,:,1);
y = z(:,:,2);
norm = sqrt(max(x.*x+y.*y,1));
z(:,:,1) = x./norm;
z(:,:,2) = y./norm;
return

function d = Dx(u)
[rows,cols] = size(u); 
d = zeros(rows,cols);
d(:,2:cols) = u(:,2:cols)-u(:,1:cols-1);
d(:,1) = u(:,1)-u(:,cols);
return

function d = Dy(u)
[rows,cols] = size(u); 
d = zeros(rows,cols);
d(2:rows,:) = u(2:rows,:)-u(1:rows-1,:);
d(1,:) = u(1,:)-u(rows,:);
return


function d = Dxt(u)
[rows,cols] = size(u); 
d = zeros(rows,cols);
d(:,1:cols-1) = u(:,1:cols-1)-u(:,2:cols);
d(:,cols) = u(:,cols)-u(:,1);
return


function d = Dyt(u)
[rows,cols] = size(u); 
d = zeros(rows,cols);
d(1:rows-1,:) = u(1:rows-1,:)-u(2:rows,:);
d(rows,:) = u(rows,:)-u(1,:);
return