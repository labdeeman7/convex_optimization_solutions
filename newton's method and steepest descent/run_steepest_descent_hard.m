% For computation define as function of 1 vector variable
F.f = @(x) 100.*(x(2) - x(1)^2).^2 + (1 - x(1)).^2; % function handler, 2-dim vector
F.df = @(x) [-400*(x(2) - x(1)^2)*x(1) - 2*(1 - x(1)); 
              200*(x(2) - x(1)^2)];  % gradient handler, 2-dim vector
F.d2f = @(x) [-400*(x(2) - 3*x(1)^2) + 2, -400*x(1); -400*x(1), 200]; % hessian handler, 2-dim vector
% For visualisation proposes define as function of 2 variables
rosenbrock = @(x,y) 100.*(y - x.^2).^2 + (1 - x).^2;

% Initialisation
alpha0 = 1;
maxIter = 1e4;
alpha_max = alpha0;
tol = 1e-6;
%=============================
% Point x0 = [1.2; 1.2]
%=============================
x0 = [-1.2; 1];

% Steepest descent line search strong WC
lsOptsSteep.c1 = 1e-4;
lsOptsSteep.c2 = 0.1;
lsFun = @(F, x_k, p_k, alpha_max, lsOptsSteep) backtracking(F, x_k, p_k, alpha_max, lsOptsSteep);
[xSteep, fSteep, nIterSteep, infoSteep] = descentLineSearch(F, 'steepest', lsFun, alpha0, x0, tol, maxIter);
% [xSteep, fSteep, nIterSteep, infoSteep] = descentLineSearch(F, 'steepest', lsFun, alpha0, x0, tol, maxIter);


%plot graph
%hard
x = linspace(0,3);
y = linspace(-5,2);
[X,Y] = meshgrid(x,y);
Z = rosenbrock(X,Y);
levels = 100:200:6000;
LW = 'linewidth'; FS = 'fontsize'; MS = 'markersize';
figure, contour(x,y,Z,levels,LW,0.5),axis([0,3 -5 2]), axis square, hold on, plot(infoSteep.xs), hold off;
figure(2)
plot(infoSteep.alphas)
figure(3)
diff = infoSteep.xs - [1;1];
norm_conv = vecnorm(diff);
plot(norm_conv);


