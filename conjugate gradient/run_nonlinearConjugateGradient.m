% For computation define as function of 1 vector variable
F.f = @(x) x(1)^2 + 5*x(1)^4 + 10*x(2)^2;
F.df = @(x) [2*x(1) + 20*x(1)^3; 20*x(2)];
F.d2f = @(x) [2 + 60*x(1)^2, 0; 0, 20];

new_func = @(x,y) x^2 + 5*x^4 + 10*y^2;

% Point
x0 = [-5; 7];
% Initialisation
alpha0 = 1;
tol = 1e-12;
maxIter = 100;

lsOptsCG_LS.c1 = 1e-4;
lsOptsCG_LS.c2 = 0.1;

lsFun = @(x_k, p_k, alpha0) lineSearch(F, x_k, p_k, alpha0, lsOptsCG_LS);

[xCG_FR_LS, fCG_FR_LS, nIterCG_FR_LS, infoCG_FR_LS] = nonlinearConjugateGradient(F, lsFun, 'FR', alpha0, x0, tol, maxIter);

x = linspace(-10,10);
y = linspace(-10,10);
[X,Y] = meshgrid(x,y);
Z = new_func(X,Y);

figure, contour(x,y,Z,10),axis([-10,10 -10 10]), axis square, hold on, plot(infoCG_FR_LS.xs, '-o'), hold off;
figure(2)
plot(infoCG_FR_LS.alphas),title('Values of line search alpha')
figure(3)
plot(infoCG_FR_LS.fs)
title('convergence of function starting point at [-5,7]')
xlabel('iterations') 
ylabel('F(x)')



