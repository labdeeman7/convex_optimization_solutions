%%%Things I have not considered 
%%%1. I have not checked if I need to define a good starting point and how
%%%for penaltry functions and augmented langrangian, 
%%%2. I need to check on the decreasing of tolerance value and the conditions
%%%said in the coursework 
%%%3. what is the correct way to change mu? 
%%%4. Ask luke about his newton method

% F.f = @(x) (x(1) - 1).^2 + 0.5.*(x(2) - 1.5).^2 - 1;
% F.df = @(x) [2*x(1) - 2; 
%               x(2) - 1.5];  % gradient handler, 2-dim vector
% F.d2f = @(x) [2 , 0;...
%               0, 1]; % hessian handler, 2-dim vector

a = 1;
b = 1.5;
mu = 1.0;

% Objective function
F.f = @(x) (x(1) - a).^2 + 0.5.*(x(2) - b).^2 - 1; % function handler, 2-dim vector
F.df = @(x) [2*(x(1) - a); 
              x(2) - b];  % gradient handler, 2-dim vector
F.d2f = @(x) [2 , 0;
              0, 1]; % hessian handler, 2-dim vector

% constraint function. since there is only one, we dont define a struct          
H.f = @(x) ((x(1).^2 + x(2).^2 - 2));
H.df = @(x) [2*(x(1)); 
              2*x(2)];
H.d2f = @(x) [2 , 0;
              0, 2];
          
% Initialisation
alpha0 = 1;
maxIter = 1000;
alpha_max = alpha0;
tol = 1e-10;
%=============================
% Point x0 = [1.2; 1.2]
%=============================
x0 = [-3; 3];

% Steepest descent line search strong WC
lsOptsSteep.c1 = 1e-4;
lsOptsSteep.c2 = 0.1;
[xPen, fPen, nIterPen, infoPen] = penalty_method(F, H, x0, mu, tol, maxIter); 



%plot graphs
%for plotting
obj_func = @(x,y) (x - a).^2 + 0.5.*(y - b).^2 - 1;

x = linspace(-5,5);
y = linspace(-5,5);
[X,Y] = meshgrid(x,y);
Z = obj_func(X,Y);

figure, contour(x,y,Z,5),axis([-5,5 -5 5]), axis square,
hold on,
plot(infoPen.xs(1,:), infoPen.xs(2, :), '-+'),
s=ezplot(@(x,y) (x).^2 + (y).^2 -2);
set(s,'linestyle',':');
title('Convergence path starting at (-3,3)')
hold off;

% figure(2)
% semilogy(infoLS_SR.xind, infoLS_SR.fs)
% figure(2)
% plot(infoPen.alphas),title('Values of line search alpha')
figure(2)
diff = infoPen.xs - [0.8571;1.1249];
norm_conv = vecnorm(diff);
plot(norm_conv);
title('convergence of function starting point at (-3,3)')
figure(3)
semilogy(norm_conv);
title('semilogy of convergence rate')



