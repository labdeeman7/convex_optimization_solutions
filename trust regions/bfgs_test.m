
% For computation define as function of 1 vector variable
F.f = @(x) (x(1) - 3*x(2)).^2 + x(1).^4;
F.df = @(x) [2*(x(1) - 3*x(2)) + 4*x(1).^3; -6*(x(1) - 3*x(2))];
F.d2f = @(x) [2 + 12*x(1).^2, -6; -6, 18];

F2 = @(x,y) (x - 3*y).^2 + x.^4;


% Starting point
x0 = [2; 2]; 

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

% Minimisation with Newton, Steepest descent and BFGS line search methods
[xLS_BFGS, fLS_BFGS, nIterLS_BFGS, infoLS_BFGS] = descentLineSearch(F, 'bfgs', lsFun, alpha0, x0, tol, maxIter)
%%

x = linspace(-11,11);
y = linspace(-11,11);
[X,Y] = meshgrid(x,y);
Z = F2(X,Y);

figure, contour(x,y,log(Z),20),axis([-11 11 -11 11]), axis square,
hold on,
plot(infoLS_BFGS.xs(1, :), infoLS_BFGS.xs(2, :), "-+"),
hold off;

figure(2)
plot(log(infoLS_BFGS.fs))
title('Line Plot of convergence of function at [10,10]')
xlabel('iterations') 
ylabel('F(x)')
