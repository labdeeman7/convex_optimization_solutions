% Rosenbrock function
F.f = @(x) 100.*(x(2) - x(1)^2).^2 + (1 - x(1)).^2; 
F.df = @(x) [-400*(x(2) - x(1)^2)*x(1) - 2*(1 - x(1)); 
              200*(x(2) - x(1)^2)];  
F.d2f = @(x) [-400*(x(2) - 3*x(1)^2) + 2, -400*x(1); -400*x(1), 200]; 
rosenbrock = @(x,y) 100.*(y - x.^2).^2 + (1 - x).^2;


%% Parameters 
% Step acceptance relative progress threshold
eta = 0.1;  
maxIter = 100; 
% Stopping tolerance on relative step length between iterations
tol = 1e-6; 


%% Trust region with 2d subspace, $x_0  = (1.2,1.2)^T$
x0 = [-1.2; 1];  
% Trust region radius
Delta = 0.2; %[0.2, 1) work well, below many iterations.

% [xTR, fTR, nIterTR, infoTR] = trustRegion(F, x0, @solverCM2dSubspaceExt, Delta, eta, tol, maxIter);

[xTR, fTR, nIterTR, infoTR] = trustRegion(F, x0, @solverCMdogleg, Delta, eta, tol, maxIter);

x = linspace(-10,10);
y = linspace(-10,10);
[X,Y] = meshgrid(x,y);
Z = rosenbrock(X,Y);

figure, contour(x,y,log(Z),20),axis([-10 10 -10 10]), axis square, hold on, plot(infoTR.xs, '-o'), hold off;

figure(2)
plot(infoTR.xind, infoTR.fs)
title('Line Plot of convergence of rosenbrock function at [1.2,1.2]')
xlabel('iterations') 
ylabel('F(x)')






