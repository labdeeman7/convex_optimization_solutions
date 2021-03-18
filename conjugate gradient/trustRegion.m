function [x_k, f_k, k, info] = trustRegion(F, x0, solverCM, Delta, eta, tol, maxIter)
% TRUSTREGION Trust region iteration
% [x_k, f_k, k, info] = trustRegion(F, x0, solverCM, Delta, eta, tol, maxIter)
% INPUTS
% F: structure with fields
%   - f: function handler
%   - df: gradient handler
%   - d2f: Hessian handler
% x_k: current iterate
% solverCM: handle to solver to quadratic constraint trust region problem
% Delta: upper limit on trust region radius
% eta: step acceptance relative progress threshold
% tol: stopping condition on minimal allowed step
%      norm(x_k - x_k_1)/norm(x_k) < tol;
% maxIter: maximum number of iterations
% OUTPUT
% x_k: minimum
% f_k: objective function value at minimum
% k: number of iterations
% info: structure containing iteration history
%   - xs: taken steps
%   - xind: iterations at which steps were taken
%   - stopCond: shows if stopping criterium was satisfied, otherwsise k = maxIter
%   
% Reference: Algorithm 4.1 in Nocedal Wright
%
% Copyright (C) 2017 Marta M. Betcke, Kiko Rullan 

% Initialization
nIter = 0;
x_k = x0;
f_0 = F.f(x0);
info.xs = x0;
info.xind = 0;
info.fs = f_0;
info.Delta = Delta;
info.ps = [0;0];
info.stopCond = false;
Delta_bound = 1;

% Loop until convergence or maximum number of iterations
while (~info.stopCond && nIter <= maxIter)
    
    % ====================== YOUR CODE HERE ===================================
    % Instructions: x_k contains the current iteration point    - used in the stopping condition
    %               x_k_1 contains the previous iteration point - used in the stopping condition
    p_k = solverCM(F,x_k, Delta);
    
    m_k0 = F.f(x_k);
    g = F.df(x_k);
    B = F.d2f(x_k);
    m_k = F.f(x_k) + g'*p_k + 0.5*p_k'*B*p_k;
    
    rho_k = (F.f(x_k) - F.f(x_k+p_k))/(m_k0 - m_k);
    
    if rho_k < 0.25
        Delta = 0.25*Delta;
    else 
        if rho_k > 0.75 && abs(norm(p_k) - Delta) < tol  
            Delta = min(2*Delta, Delta_bound);
        else
            Delta = Delta;
        end
    end
    nIter = nIter + 1;
    
    if rho_k > eta
        x_k_1 = x_k;
        x_k = x_k + p_k;
        
        info.xs = [info.xs, x_k];
        info.ps = [info.ps, p_k];
        info.xind = [info.xind, nIter];
        info.fs = [info.fs, F.f(x_k)];
        normStep = norm(x_k - x_k_1)/norm(x_k);
        info.stopCond = (normStep < tol); 
        
    else
        x_k = x_k;
    end
    
   
    info.Delta = [info.Delta, Delta];
    
end
f_k = F.f(x_k);
k = nIter;

end