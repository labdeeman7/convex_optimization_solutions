
% For computation define as function of 1 vector variable
F.f = @(x) (x(1) - 3*x(2)).^2 + x(1).^4;
F.df = @(x) [2*(x(1) - 3*x(2)) + 4*x(1).^3; -6*(x(1) - 3*x(2))];
F.d2f = @(x) [2 + 12*x(1).^2, -6; -6, 18];
testFxn = @(x,y) (x - 3*y).^2 + x.^4;
% Starting point
x0 = [10; 10]; 

% Parameters
maxIter = 200; 
tol = 1e-10; % Stopping tolerance on relative step length between iterations

% Line search parameters
alpha0 = 1;

% Strong Wolfe LS
lsOpts_LS.c1 = 1e-4;
lsOpts_LS.c2 = 0.2; % 0.1 Good for Newton, 0.9 - good for steepest descent, 0.5 compromise.
lsFunS = @(x_k, p_k, alpha0) lineSearch(F, x_k, p_k, alpha0, lsOpts_LS);
lsFun = lsFunS;


tic;
% Minimisation with Newton, Steepest descent and BFGS line search methods
[xLS_BFGS, fLS_BFGS, nIterLS_BFGS, infoLS_BFGS] = descentLineSearch(F, 'bfgs', lsFun, alpha0, x0, tol, maxIter)
toc







pathD = [];
for i = 1:size(infoLS_BFGS.xs,2)
pathDtemp = F.f(infoLS_BFGS.xs(:,i));
pathD = [pathD; pathDtemp];
end
figure,
plot(1:length(infoLS_BFGS.alphas),infoLS_BFGS.alphas)
title('Step sizes used by BFGS algorithms')


figure,
plot3(infoLS_BFGS.xs(1,:),infoLS_BFGS.xs(2,:),pathD.','r','LineWidth',2)
hold on 
title('trajectories traced by the iterates by BFGS algorithms')


n = 300;
x1 = linspace(10,0,n+1);
x2 = linspace(10,0,n+1);
[X,Y] = meshgrid(x1,x2);
surfc(X, Y, testFxn(X,Y), 'EdgeColor', 'none')
%contourf(X, Y, testFxn(X,Y))


error = [];
for i = 1:size(infoLS_BFGS.xs,2)-1
    errorTemp = norm(infoLS_BFGS.xs(:,i+1)-infoLS_BFGS.xs(:,nIterLS_BFGS));
    error = [error, errorTemp];
end
figure,
plot(1:length(error),error)
title('Rate of convergence by BFGS algorithms')


figure,
plot(1:length(infoLS_BFGS.error),infoLS_BFGS.error)
title('BFGS error in hessian approximation')









function [xMin, fMin, nIter, info] = descentLineSearch(F, descent, ls, alpha0, x0, tol, maxIter)
% DESCENTLINESEARCH Wrapper function executing  descent with line search
% [xMin, fMin, nIter, info] = descentLineSearch(F, descent, ls, alpha0, x0, tol, maxIter) 
%
% INPUTS
% F: structure with fields
%   - f: function handler
%   - df: gradient handler
%   - d2f: Hessian handler
% descent: specifies descent direction {'steepest', 'newton', 'bfgs'}
% ls: specifies line search algorithm
% alpha0: initial step length 
% x0: initial iterate
% tol: stopping condition on minimal allowed step
%      norm(x_k - x_k_1)/norm(x_k) < tol;
% maxIter: maximum number of iterations
%
% OUTPUTS
% xMin, fMin: minimum and value of f at the minimum
% nIter: number of iterations 
% info: structure with information about the iteration 
%   - xs: iterate history 
%   - alphas: step lengths history 
%
% Copyright (C) 2017  Marta M. Betcke, Kiko Rullan

% Parameters
% Stopping condition {'step', 'grad'}
stopType = 'grad';

% Extract inverse Hessian approximation handler
extractH = 1;

% Initialization
nIter = 0;
x_k = x0;
info.xs = x0;
info.alphas = alpha0;
info.error = [];
stopCond = false; 
I = eye(2);
switch lower(descent)
  case 'bfgs'
    H_k = I;
    % Store H matrix in columns
    info.H = [];
end

% Loop until convergence or maximum number of iterations
while (~stopCond && nIter <= maxIter)
    
  % Increment iterations
    nIter = nIter + 1;

    % Compute descent direction
    switch lower(descent)
      case 'steepest'
        p_k = -F.df(x_k); % steepest descent direction
      case 'newton'
        p_k = -F.d2f(x_k)\F.df(x_k); % Newton direction
        if p_k'*F.df(x_k) > 0 % force to be descent direction (only active if F.d2f(x_k) not pos.def.)
          p_k = -p_k;
        end
      case 'bfgs'
        %======================== YOUR CODE HERE ==========================================
        p_k =  -H_k*F.df(x_k); % BFGS direction

        %==================================================================================

    end
    
    % Call line search given by handle ls for computing step length
    alpha_k = ls(x_k, p_k, alpha0);
    
    % Update x_k and f_k
    x_k_1 = x_k;
    x_k = x_k + alpha_k*p_k;
    
    switch lower(descent)
      case 'bfgs'
          
        %======================== YOUR CODE HERE ==========================================
         s_k = x_k-x_k_1;
         y_k = F.df(x_k)-F.df(x_k_1);




        %==================================================================================
        
        if nIter == 1
%           Update initial guess H_0. Note that initial p_0 = -F.df(x_0) and x_1 = x_0 + alpha * p_0.
          disp(['Rescaling H0 with ' num2str((s_k'*y_k)/(y_k'*y_k)) ])
          H_k =   (s_k'*y_k)/(y_k'*y_k) * I;
        end
        
        
        %======================== YOUR CODE HERE ==========================================
        H_k = (I-(s_k*y_k')/(y_k'*s_k)) * H_k *(I-(y_k*s_k')/(y_k'*s_k))+(s_k*s_k')/(y_k'*s_k);



        %==================================================================================
        
        if extractH
            % Extraction of H_k as handler
            info.H{length(info.H)+1} = H_k;
        end
    end
    info.error = [info.error;norm(eye(2)-H_k*F.d2f((x_k)))]
    
    

    % Store iteration info
    info.xs = [info.xs x_k];
    info.alphas = [info.alphas alpha_k];
    
    switch stopType
      case 'step' 
        % Compute relative step length
        normStep = norm(x_k - x_k_1)/norm(x_k_1);
        stopCond = (normStep < tol);
      case 'grad'
        stopCond = (norm(F.df(x_k), 'inf') < tol*(1 + abs(F.f(x_k))));
    end
    
end

% Assign output values 
xMin = x_k;
fMin = F.f(x_k); 
end