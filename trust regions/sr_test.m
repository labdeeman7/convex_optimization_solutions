
% For computation define as function of 1 vector variable
F.f = @(x) (x(1) - 3*x(2)).^2 + x(1).^4;
F.df = @(x) [2*(x(1) - 3*x(2)) + 4*x(1).^3; -6*(x(1) - 3*x(2))];
F.d2f = @(x) [2 + 12*x(1).^2, -6; -6, 18];

F2 = @(x,y) (x - 3*y).^2 + x.^4;  % for visualization

% Starting point
x0 = [10; 10]; 

% Parameters
maxIter = 200; 
tol = 1e-10; % Stopping tolerance on relative step length between iterations
Delta = 0.2; 
eta = 0.1;  

% Stopping tolerance on relative step length between iterations
% tol = 1e-6;

% Line search parameters
alpha0 = 1;

% Strong Wolfe LS
lsOpts_LS.c1 = 1e-4;
lsOpts_LS.c2 = 0.2; % 0.1 Good for Newton, 0.9 - good for steepest descent, 0.5 compromise.
lsFunS = @(x_k, p_k, alpha0) lineSearch(F, x_k, p_k, alpha0, lsOpts_LS);
lsFun = lsFunS;

debug = false;

% Minimisation with Newton, Steepest descent and BFGS line search methods
[xLS_SR, fLS_SR, nIterLS_SR, infoLS_SR] = trustRegionOld(F, x0, @solverCM2dSubspaceExt,Delta, eta, tol, maxIter, debug, F2)



x = linspace(-11,11);
y = linspace(-11,11);
[X,Y] = meshgrid(x,y);
Z = F2(X,Y);

figure, contour(x,y,log(Z),20),axis([-11 11 -11 11]), axis square,
hold on,
plot(infoLS_SR.xs(1, :), infoLS_SR.xs(2, :), "-+"),
hold off;

figure(2)
semilogy(infoLS_SR.xind, infoLS_SR.fs)
% plot(infoLS_SR.xind, infoLS_SR.fs)

% figure(3)
% diff = infoLS_SR.xs - [0;0];
% norm_conv = vecnorm(diff);
% % plot(norm_conv);
% % semilogy(infoLS_SR.xind, norm_conv)


title('Line Plot of convergence of function at [10,10]')
xlabel('iterations') 
ylabel('F(x)')







