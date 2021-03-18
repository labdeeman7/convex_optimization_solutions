% For computation define as function of 1 vector variable
F.f = @(x) (x(1) - 3*x(2)).^2 + x(1).^4;
F.df = @(x) [2*(x(1) - 3*x(2)) + 4*x(1).^3; -6*(x(1) - 3*x(2))];
F.d2f = @(x) [2 + 12*x(1).^2, -6; -6, 18];
testFxn = @(x,y) (x - 3*y).^2 + x.^4;                 
% Parameters
maxIter = 200; 
tol = 1e-10; % Stopping tolerance on relative step length between iterations
debug = 0; % Debugging parameter will switch on step by step visualisation of quadratic model and various step options

% Starting point
x0 = [10; 10]; 

% Trust region parameters 
eta = 0.1;  % Step acceptance relative progress threshold
Delta = 0.9; % Trust region radius

% Minimisation with 2d subspace and dogleg trust region methods
Fsr1 = rmfield(F,'d2f');
tic;
[xTR_SR1, fTR_SR1, nIterTR_SR1, infoTR_SR1] = trustRegion(Fsr1, x0, @solverCM2dSubspaceExt, Delta, eta, tol, maxIter, debug);
toc




pathD = [];
for i = 1:size(infoTR_SR1.xs,2)
pathDtemp = F.f(infoTR_SR1.xs(:,i));
pathD = [pathD; pathDtemp];
end
% figure,
% plot(1:length(infoTR_SR1.alphas),infoTR_SR1.alphas)
% title('Step sizes used by PR CG algorithms')


figure,
plot3(infoTR_SR1.xs(1,:),infoTR_SR1.xs(2,:),pathD.','r','LineWidth',2)
hold on 
title('trajectories traced by the iterates by SR1 algorithms')


n = 300;
x1 = linspace(10,0,n+1);
x2 = linspace(10,0,n+1);
[X,Y] = meshgrid(x1,x2);
surfc(X, Y, testFxn(X,Y), 'EdgeColor', 'none')
%contourf(X, Y, testFxn(X,Y))


convergence = [];
for i = 1:size(infoTR_SR1.xs,2)-1
    errorTemp = norm(infoTR_SR1.xs(:,i+1)-infoTR_SR1.xs(:,nIterTR_SR1));
    convergence = [convergence, errorTemp];
end
figure,
plot(1:length(convergence),convergence)
title('Rate of convergence by SR1 algorithms')


figure,
plot(1:length(infoTR_SR1.error),infoTR_SR1.error)
title('Error in hessian approximation SR1')





