% Perform image segmentation using adaptive PDHG.  The problem solved is
%            min_u    TV(u) + mu <u, (im-c1)^2-(im-c2)^2 >
%  Inputs...
%    im    : An 2D array containing the image
%    mu    : A scalar, the scale parameter for the segmentation
%    c1    : The average pixel intensity inside region 1 (scalar double)
%    c2    : The average pixel intensity inside region 2 (scalar double)
%    opts  : An optional argument for customizing the behavior of 
%              pdhg_adaptive
%  Outputs...
%    segment :  A vector of values in [0,1]. Threshold this vector at some
%               level between 0 and 1 to obtain a segmentation.
%    out      : A struct containing convergence data generated by the
%                  function 'pdhg_adaptive' when it solved the problem.
%
%  This File requires that 'pdhg_adaptive.m' be in your current path. 
%
%  For an explanation of PDHG and how it applies to this problem, see
%    "Adaptive Primal-Dual Hybrid Gradient Methods for Saddle-Point
%    Problems"  available at <http://arxiv.org/abs/1305.0546>
%

function [ segment, out ] = pdhg_segment( im, mu, c1, c2, opts )

  %  Define a vector that is positive for pixels close to c2, and negative
  %  for pixels close to c1.
    f = (im-c1).^2-(im-c2).^2;
    f = f*mu;
    
    %  Prox operator of primal variables.  Performs gradient descent on x,
    % and reproject back into the unit iterval
    fProx = @(x,tau) min(max(x-tau*f,0),1); 
    %  Project dual variables onto the unit ball
    gProx = @(y,sigma) projectInf(y);
    % gradient and divergence operators
    A = @(x) grad(x);
    At = @(y) -div(y);

    %  Initial guess
    [rows,cols] = size(im);
    x0 = zeros(size(im));
    y0 = zeros(rows,cols,2);
    
    %  Options for the solver
     if ~exist('opts','var')
        opts = [];
    end
   
    %  f1 computes the objective value of iterates
    % opts.f1 = @(x,y,x0,y0,tau,sigma) sum(sum(sqrt(Dx(x).^2+Dy(x).^2)))+sum(sum(x.*f));

    [segment ,out]= pdhg_adaptive(x0,y0,A,At,fProx,gProx, opts);    

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
