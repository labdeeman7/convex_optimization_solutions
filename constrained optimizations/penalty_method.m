function [xMin, fMin, nIter, infoPenalty] = penalty_method(F, H, x0, mu, tol, maxIter)
% penalty_method function to minimise a quadratic form with constraints
%
% INPUTS
% F and H: structure with fields
%  - f: function to minimise
%  - df: gradient of function 
%  - d2f: Hessian of function 
% x0: initial iterate
% mu: penalty factor
% tol: tolerance
% maxIter: maximum number of iterations
%
% OUTPUTS
% xMin, fMin: minimum and value of f at the minimum
% nIter: number of iterations 
% infoPenalty: structure with information about the iteration 
%   - xs: iterate history for x 
%   - ys: iterate history for y


% Initilize
nIter = 0;
stopCond = false;
x_k = x0;
infoPenalty.xs = x_k;
infoPenalty.fs = F.f(x_k);
infoPenalty.hs = H.f(x_k);
infoPenalty.inIter = 0;

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
    % Create function handler for centering step (needs to be redifined at each step because of changing "t")
    Q.f = @(x) F.f(x) + 0.5*mu*(H.f(x)).^2;
    Q.df = @(x) F.df(x) + mu*H.f(x)*H.df(x);
    Q.d2f = @(x) F.d2f(x) + mu*H.f(x)*H.d2f(x) + mu*H.df(x)*H.df(x)';
    
    % unconstrained minimimzation
    lsFun = @(Q, x_k, p_k, alpha0, opts) backtracking(Q, x_k, p_k, alpha0, opts);
    x_k_prev = x_k;
    [x_k, f_k, nIterLS, infoIter] = descentLineSearch(Q, 'Newton', lsFun, alpha0, x_k, tolNewton, maxIterNewton);     
    h_k = H.f(x_k);
    
    % Check stopping condition.
    if norm(x_k - x_k_prev) < tol; stopCond = true; end
    disp(norm(x_k - x_k_prev))
    
    % Increase mu
    mu = mu*1.5;     
    
    % Store info
    infoPenalty.xs = [infoPenalty.xs x_k];
    infoPenalty.fs = [infoPenalty.fs f_k];
    infoPenalty.hs = [infoPenalty.hs h_k];
    infoPenalty.inIter = [infoPenalty.inIter nIterLS];
    
    % Increment number of iterations
    nIter = nIter + 1;
end

% Assign values
xMin = x_k;
fMin = F.f(x_k);

