function [xMin, fMin, nIter, infoAugmented] = augmented_lagrangian(F, H, x0, mu,nu, tol, maxIter)
% penalty_method function to minimise a quadratic form with constraints
% INPUTS
% F and H: structure with fields
%  - f: function to minimise
%  - df: gradient of function 
%  - d2f: Hessian of function 
% x0: initial iterate
% mu: penalty factor
% nu: langrangian multiplier
% tol: tolerance
% maxIter: maximum number of iterations
%
% OUTPUTS
% xMin, fMin: minimum and value of f at the minimum
% nIter: number of iterations 
% infoAugmented: structure with information about the iteration 
%   - xs: iterate history for x 
%   - ys: iterate history for y
%
% Copyright (C) 2017  Kiko Rul·lan, Marta M. Betcke

% Initilize
nIter = 0;
stopCond = false;
x_k = x0;
infoAugmented.xs = x_k;
infoAugmented.fs = F.f(x_k);
infoAugmented.hs = H.f(x_k);
infoAugmented.nu = nu;
infoAugmented.inIter = 0;

% Parameters for centering step
alpha0 = 1; 
opts.c1 = 1e-4;
opts.c2 = 0.9;
opts.rho = 0.5;
tolNewton = 1e-12;
maxIterNewton = 100;
% Loop 
while (~stopCond && nIter < maxIter)
    disp(strcat('Iteration ', int2str(nIter)));
    % Merit function
    Q.f = @(x) F.f(x) + nu*H.f(x) + 0.5*mu*(H.f(x)).^2;
    Q.df = @(x) F.df(x) + nu*H.df(x) + mu*H.f(x)*H.df(x);
    Q.d2f = @(x) F.d2f(x) + nu*H.d2f(x) + mu*H.f(x)*H.d2f(x) + mu*H.df(x)*H.df(x)';
    
    % Line search function (needs to be redefined at each step because of changing G) 
    %lsFun = @(x_k, p_k, alpha0) lineSearch(G, x_k, p_k, alpha0, opts);
    lsFun = @(Q, x_k, p_k, alpha0, opts) backtracking(Q, x_k, p_k, alpha0, opts);
    
    % unconstrained minimization
    x_k_prev = x_k;
    [x_k, f_k, nIterLS, infoIter] = descentLineSearch(Q, 'Newton', lsFun, alpha0, x_k, tolNewton, maxIterNewton);     
    h_k = H.f(x_k);
    
    % update nu 
    nu = nu + (mu*h_k); 
    
    % Check stopping condition.
    if norm(x_k - x_k_prev) < tol; stopCond = true; end
    disp(norm(x_k - x_k_prev))   
    
    % Store info
    infoAugmented.xs = [infoAugmented.xs x_k];
    infoAugmented.fs = [infoAugmented.fs f_k];
    infoAugmented.hs = [infoAugmented.hs h_k];
    infoAugmented.nu = [infoAugmented.nu nu];
    infoAugmented.inIter = [infoAugmented.inIter nIterLS];
   
    % Increment number of iterations
    nIter = nIter + 1;
end

% Assign values
xMin = x_k;
fMin = F.f(x_k);

