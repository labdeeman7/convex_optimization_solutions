function p = solverCMdogleg(F, x_k, Delta)
% SOLVERCMDOGLEG Solves quadratic constraint trust region problem via 2d subspace
% p = solverCMdogleg(F, x_k, Delta)
% INPUTS
% F: structure with fields
%   - f: function handler
%   - df: gradient handler
%   - d2f: Hessian handler
% x_k: current iterate
% Delta: upper limit on trust region radius
% OUTPUT
% p: step (direction times lenght)
%
% Copyright (C) 2017 Marta M. Betcke, Kiko Rul·lan 

%% Calculate p_u and p_b
p_u = -( (F.df(x_k)' * F.df(x_k)) / (F.df(x_k)' * F.d2f(x_k) * F.df(x_k)) ) * F.df(x_k);

p_b = -inv(F.d2f(x_k)) * F.df(x_k); 

%% Conditions, dogleg path in trust region, intersects first leg, and intersects second leg
if norm(p_b) <= Delta
    p = p_b;
elseif norm(p_u) >= Delta
   p = Delta * (p_u) / norm(p_u);
else 
    %% solving for tau, first solve or tau-1, then transfer -1 over
    a = (p_b'*p_b) - 2*(p_b'*p_u) + (p_u'*p_u);
    b = 2*(p_b'*p_u) - 2*(p_u'*p_u);
    c = (p_u'*p_u) - Delta^2;
    
    r = roots([a,b,c]);
    taus = r+1;
    
    tau_in_range_index = find(taus >= 1 & taus <= 2);
    tau_in_range = taus(tau_in_range_index(1));
    
    p = p_u + (tau_in_range - 1) * (p_b - p_u);
end



end
