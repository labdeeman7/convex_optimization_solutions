function [u, v, w, iter, stopValue, uvIterates, info] = ADMM_TV(f, A , Atr, invLS, E, Etr, F, b, Proxy, u0, para)
% ADMM_TV implements the alternating direction method of multipliers to solve 
%   min obj(x) = 1/2 * || A u - f ||_2^2 + J(u)
% by introducing a split of u as  
%   E u + F v = b
% and solving 
%   1/2 * || A u - f ||_2^2 + J(v)   subject to   E u + F v = b
% by alternating updates of u, v and w, where J is the anoistorpic total
% variation,i.e. J(u) = ||Du||_1
% see Boyd et al., 2011, "Distributed Optimization and Statistical Learning 
% via the Alternating Direction Method of Multipliers"
%
% Note: This code is designed for large-scale, matrix-free problems and will perform poorly on others! 
%
%  INPUT:
%   A           - function handle for y = A(x) 
%   Atr         - function handle for x = A^T(y) 
%   f           - the data f
%   invLS       - function handle to the fully quadratic subproblem (involving the smooth part of the objective)
%                     argmin || A u - f ||_2^2 + rho || w - v ||_2^2
%                 This can be frequently made much more efficient for particular problem structure.
%                 If empty use generic LS solve.
%   E           - function handle for the split operator E
%   Etr         - function handle for the adjoint to split operator E
%   F           - function handle for the split operator F
%   b           - right hand side of the split equation
%   Proxy       - proximal operator as a function handle of x and alpha to solve 
%                   prox_{J,lambda}(x) = argmin_z ( J(z) + 1/(2*lambda) ||x - F z ||_2^2   ) 
%   u0          - a start value for x
%   para        - a struct containing all optional parameters:
%     'rho' - quadratic penalty parameter weighing the primal constraint rho/2 * || E u + F v - b ||^2          
%     'overRelaxPara' - over-relaxation parameter (see Section 3.4.3. in Boyd et al, 2011),
%                       default: 1.8, i.e., overrelaxation is in use           
%     'stopTolerance' - stop tolerance when both primal and dual residual are small enough (see 3.3.1. in Boyd et al.)   
%     'maxIter' - maximum number of iteration after which to stop IN ANY CASE
%                (even if other convergence criteria are not met yet)
%
%  OUTPUTS:
%   u           - first version of the primary variable
%   v           - second version of the primary variable (might be preferable to use 
%                 as it fulfills all hard constraints and is usually thresholded)
%   w           - dual variable
%   iter        - number of iterations
%   stopValue   - stop value (interesting if fixed number of iterations is used)  
%   uvIterates  - optionally, u(1:returnIteratesInc:end) and v(1:returnIteratesInc:end) will be returned
%                 as well
%   info        - information about the iteration
%
% Copyright (C) 2015 Felix Lucka (PAT Toolbox) 
%                    modified by Bolin Pan, Kiko Rul·lan and Marta M. Betcke

%========================================
% Read out parameters (see above)
%========================================

% Stop Tolerance
if (isfield(para, 'stopTolerance'))
    stopTolerance = para.stopTolerance;
else 
    disp('Default Stop Tolerance = 1e-6');
    stopTolerance = 1e-6;
end
% Maximum number of iterations
if (isfield(para, 'maxIter'))
    maxIter = para.maxIter;
else 
    disp('Default number of iterations = 100');
    maxIter = 100;
end
% Rho
if (isfield(para, 'rho'))
    rho = para.rho;
else 
    disp('Default rho = 1');
    rho = 1;
end
% Mu
if (isfield(para, 'mu'))
    mu = para.mu;
else 
    disp('Default mu = 10');
    mu = 10;
end
% Tau
if (isfield(para, 'tau'))
    tau = para.tau;
else 
    disp('Default tau = 2');
    tau = 2;
end
% Rho min
if (isfield(para, 'rhoMin'))
    rhoMin = para.rhoMin;
else 
    disp('Default rho min = 1e-5');
    rhoMin = 1e-5;
end
% Over Relaxation Parameter
if (isfield(para, 'overRelaxPara'))
    overRelaxPara = para.overRelaxPara;
else 
    disp('Default over relaxation parameter = 1.8');
    overRelaxPara = 1.8;
end
%========================================
% Initialize inner variables
%========================================
u    = u0;
v    = 0*u0;
v1   = v;
v2   = v;
E1   = E{1};
E2   = E{2};
Etr1 = Etr{1};
Etr2 = Etr{2};
w1   = E1(u)+F(v1)-b;
w2   = E2(u)+F(v2)-b;
Fv1  = F(v1);
Fv2  = F(v2);
iter = 0;
stopValue       = Inf;
stopFL          = false; 
uvIterates      = {};

N = length(u0); %size of the solutiong
M = length(f);  %size of the data

epsAbs = stopTolerance;
epsRel = stopTolerance;
stopTolerance = 1;

info = [];

disp('Starting ADMM iterations...')
    
%========================================
% START THE ADMM ITERATION 
%========================================
while(~stopFL && iter < maxIter)
    % Proceed with the iteration
    iter = iter + 1;

    % Update u: argmin || A u - f ||_2^2 + rho || u - w + v ||_2^2    
    if isa(invLS,'function_handle')
      auxVar = Atr(f) + rho*(v - w);
      u = invLS(A, Atr, auxVar, rho);     
    else
      tolLS = epsRel^(1/2);
      maxItLS = 200;
      RHS = reshape(Atr(f),M,M) + rho*Etr1(v1 - w1) + rho*Etr2(v2 - w2);
      LHS = @(x) Atr(A(x)) + rho*reshape(Etr1(E1(reshape(x,M,M))),[],1) + rho*reshape(Etr2(E2(reshape(x,M,M))),[],1);
      % Solve the corresponding normal equations
      [u,flag,relres,iterPCG,resvec] = pcg(LHS, RHS(:), tolLS, maxItLS);
    end    
    
    Eu1 = reshape(E1(u),M,M);
    Eu2 = reshape(E2(u),M,M);
    
    % Over-relaxation 
    uRel1 = overRelaxPara * Eu1 - (1-overRelaxPara) * (Fv1 - b);
    uRel2 = overRelaxPara * Eu2 - (1-overRelaxPara) * (Fv2 - b);
    % Update v: argmin tau*||v||_1 + rho/2|| u - b + w ||_2^2
    v1Old = v1;
    v2Old = v2;
    v1  = Proxy(uRel1 - b + w1, 1/rho);
    v2  = Proxy(uRel2 - b + w2, 1/rho);
    Fv1 = F(v1);
    Fv2 = F(v2);
    % Update dual variable w
    w1 = w1 + (uRel1 + Fv1 - b);
    w2 = w2 + (uRel2 + Fv2 - b);

    % Update primal and dual residuum
    primalRes1      = Eu1 + Fv1 - b;
    primalRes2      = Eu2 + Fv2 - b;
    primalRes       = [primalRes1(:); primalRes2(:)];
    primalResNorm   = norm(primalRes);
    dualRes1        = rho * Etr1(F(v1 - v1Old));
    dualRes2        = rho * Etr2(F(v2 - v2Old));
    dualRes         = [dualRes1(:); dualRes2(:)];
    dualResNorm     = norm(dualRes);

    % Check stop conditions
    epsPrimal = sqrt(numel(primalRes)) * epsAbs + epsRel * max(max(norm([Eu1(:); Eu2(:)]),norm([Fv1(:); Fv2(:)])), norm(b(:)));
    epsDual   = sqrt(numel(dualRes)) * epsAbs + epsRel * rho * (norm([reshape(Etr1(w1),[],1);reshape(Etr2(w2),[],1)]));
    stopValue = max(primalResNorm/epsPrimal,dualResNorm/epsDual);
    stopFL = stopValue < 1;
    stopT(iter) = stopValue;
    
    % Update rho
    if(primalResNorm > mu * dualResNorm)
        rho = tau * rho;
        w1 = w1/tau;
        w2 = w2/tau;
    elseif(dualResNorm > mu * primalResNorm && rho/tau > rhoMin)
        rho = rho/tau;
        w1 = tau * w1;
        w2 = tau * w2;
    end
    info.rho(iter) = rho;

    % Output
    outputStr =  ['Iteration ' int2str(iter) ';   stop value / stop tolerance: ' num2str(stopValue/stopTolerance,'%.6e')];
    disp(outputStr);
    % store iterates
    uvIterates{end+1}.u    = u;
    uvIterates{end}.v{1}   = v1;
    uvIterates{end}.v{2}   = v2;
end

outputStr =  ['ADMM ended at iteration ' int2str(iter) ];
disp(outputStr);

% return some information
info.iter = iter;
info.stopT = stopT;

% return the auxilary variables
v = [];
v{1} = v1;
v{2} = v2;

% return dual variables
w{1} = w1;
w{2} = w2;

end

 