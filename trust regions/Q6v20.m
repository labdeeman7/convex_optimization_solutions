clc; clear;
t = [4/200:4/200:4];

%% samples data
f =  @(x,t) ( x(1) + x(2)*t^2) * exp( -x(3)* t );
maxAmp = 20.7146;
var = 0.05 * maxAmp;
org = [];
samples = [];
x0 = [3;150;2];
for i = 1:200
    org = [org, f( x0,t(i) )];
    samples = [samples, f(x0,t(i))+ normrnd(0,var)];
%     samples = [samples, f(x0,t(i))];
end
%% R function:
F.f = @(x) sum( (( x(1) + x(2)*t.^2) .* exp( -x(3)* t ) - samples).^2 );
F.df = @(x) sum( [ 2*( (x(1)+x(2)*t.^2).*exp(-x(3)*t) - samples ).*exp(-x(3) * t); ....
                   2*( (x(1)+x(2)*t.^2).*exp(-x(3)*t) - samples ).*(t.^2 .* exp(-x(3) .* t)) ;....
                   2*( (x(1)+x(2)*t.^2).*exp(-x(3)*t) - samples ).*(-t*x(1).*exp(-x(3)*t) - t.^3 .* x(2).*exp(-x(3)*t)) ],2);%3x1? suppose to be 1x3??            
F.r = @(x)  [ (( ( x(1) + x(2)*t.^2) .* exp( -x(3)*t )) - samples ) ]';
F.J = @(x)  [    exp(-x(3)*t); ....
                (t.^2 .* exp(-x(3)*t)) ;....
                (-t*x(1).*exp(-x(3)*t) - t.^3.*x(2).* exp(-x(3)*t)) ]'; 
F.d2r = @(x) [ 0*t,...
               0*t,...
               -1*t.*exp(-x(3)*t);...
               0*t,...
               0*t,...
               -1*t.^3.*exp(-x(3)*t);...
               -1*t.*exp(-x(3)*t),...      
               -1*t.^3.*exp(-x(3)*t),...       
               x(1)*t.^2.*exp(-x(3)*t)+x(2)*t.^4.*exp(-x(3)*t);];          

%% op
alpha0 = 1;
lsOpts_LS.c1 = 1e-4;
lsOpts_LS.c2 = 0.2;
lsFunS = @(x_k, p_k, alpha0) lineSearch(F, x_k, p_k, alpha0, lsOpts_LS);
lsFun = lsFunS;
x0 = [1 ; 1 ; 1];
tol = 1e-10;
maxIter = 10000;
[xLS, nIterLS, infoLS] = descentLineSearch2(F, lsFun, alpha0, x0, tol, maxIter);
%% plot the result
res = [];
for i = 1:200
    res = [res, f( xLS,t(i) )];
end
disp( sqrt(F.f(xLS)/200 ) ) 
scatter(t,samples);

hold on;
plot(t,res);

hold on;
plot(t,org);


error = [];
for i = 1:size(infoLS.xs,2)-1
    errorTemp = norm(infoLS.xs(:,i+1)-infoLS.xs(:,nIterLS));
    error = [error, errorTemp];
end
figure,
semilogy(1:length(error),error)
title('Convergence plot of GN')


% 
% error = zeros(1,nIterLS);
% sum = zeros(3,3);
% rHessain = [];
% for i=1:nIterLS
%     rHessain = F.d2r(infoLS.xs(:,i));
%     r = F.r(infoLS.xs(:,i));
%     for j = 1:200
%         sum = r(j) * rHessain(:,3*j-2:3*j);
%     end
%     H = F.J(infoLS.xs(:,i))'*F.J(infoLS.xs(:,i)) + sum;
%     e = norm(inv(F.J(infoLS.xs(:,i))'*F.J(infoLS.xs(:,i))*H));
%     error(i) = e;
% end
% 
% plot(1:nIterLS,error);

r = F.r(xLS);
H = 0;
rHessain = F.d2r(xLS);
    for j = 1:200
        H = H + r(j) * rHessain(:,3*j-2:3*j);
    end
J_x = F.J((xLS));
factor = norm(inv(J_x'*J_x)*H)
%% function handle 
function [xMin, nIter, info] = descentLineSearch2(F,ls, alpha0, x0, tol, maxIter)
% Initialization
nIter = 0;
normError = 1;
x_k = x0;
info.xs = x0;
info.alphas = alpha0;
t = [4/200:4/200:4];
% Loop until convergence or maximum number of iterations
while (normError >= tol && nIter <= maxIter)
    
  % Increment iterations
    nIter = nIter + 1;
      %================================= YOUR CODE HERE =============================
      p_k = -inv(F.J(x_k)'*F.J(x_k) )* F.J(x_k)' * F.r(x_k);
      %==============================================================================
    % Call line search given by handle ls for computing step length
    alpha_k = ls(x_k, p_k, alpha0);
    
    % Update x_k and f_k
    x_k_1 = x_k;
    x_k = x_k + alpha_k*p_k;
    % Compute relative error norm
    normError = norm(x_k - x_k_1)/norm(x_k_1); 
    % Store iteration info
    info.xs = [info.xs,x_k];
    info.alphas = [info.alphas alpha_k];
end
    
% Assign output values 
xMin = x_k;
disp(nIter);
end


