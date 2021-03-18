%% Assignment 4 Question 4
clc;clear all;close all;
% Set states for repeatable experiments
rng(12345);

% Generate 'Ground Truth'
spikes = datasample(1:2^13, 100, 'Replace', false);
x = zeros(2^13,1);
for i = 1:100
    u = rand(1);
    if u < 0.5
        x(spikes(i))=1;
    else
        x(spikes(i))=-1;
    end
end



%% (1.1) Gaussian random matrix A + ISTA
N = 2^13; K = 2^10;
A_matrix = orth(randn(N)); 
r = datasample(1:N, K, 'Replace', false);
A_matrix = A_matrix(r,:);
% Then generate y
sigma = 0.005; %e = normrnd(0,0.005,K,1);
y = A_matrix*x;
y = y + sigma*max(abs(y))*randn(K,1);
%y = A*x + max(abs(A*x))*e;

%==================================================
% Initialization ISTA/FISTA ADMM
%==================================================
A = @(x) A_matrix * x;
At = @(x) A_matrix' * x;
AtA = @(x) At(A(x));
% Initial guess: 
x0 = At(y);  % min L2-norm (energy) solution
%x0 = zeros(N,1); % all 0s
% Regularization parameter
lambda = 0.1*max(abs(x0));

%==================================================
% FUNCTION DEFINITION ISTA/FISTA
%==================================================
% L2 Data fit term 
F.f = @(x,y) 0.5*sum((A(x) - y).^2); 
% Gradient of the data fit term
F.df = @(x,y) At(A(x) - y); 
% L1 regularization term
G.f = @(x) norm(x,1);
% Proximal operator
G.prox = @(x, alpha) softThresh(x, alpha); 
% Compute the Liptchitz constant of AtA
L = 1;
% Objective function as in eg gradient descent 
obj.f = @(x,lambda,y) F.f(x,y) + lambda*G.f(x); % Regularized objective function
obj.L = L; % Lipschitz constant for the gradient
%==================== 
% ISTA 
%==================== 
x0 = zeros(N,1);
[fISTA, infoISTA.obj, infoISTA.mse,infoISTA.stopC] = ista(y, F, G, lambda, x0, 1, 1e-5, 200, x, @(x) x);
[infoISTA.obj(end) infoISTA.mse(end)]
MSEista = infoISTA.mse;



length(find(fISTA>0.1))
length(find(fISTA<-0.1))
x_ISTA = zeros(N,1);
x_ISTA(find(fISTA>0.1)) = 1;
x_ISTA(find(fISTA<-0.1)) = -1;
plot(x); hold on; plot(x_ISTA)
xlabel('n'); ylabel('amplitude')
length(intersect(find(x==1),find(x_ISTA==1)))+length(intersect(find(x==-1),find(x_ISTA==-1)))









%% (1.2) Gaussian random matrix A + FISTA
%==================== 
x0 = At(y);  % min L2-norm (energy) solution
%x0 = zeros(N,1); % all 0s
% Regularization parameter
lambda = 0.1*max(abs(x0));
x0 = zeros(N,1);
[fNEST, infoNEST.obj, infoNEST.mse] = fista(y, F, G, lambda, x0, 1, 1e-5, 100, x, @(x) x);
MSEfista = infoNEST.mse;
infoNEST.obj

length(find(fNEST>0.1));
length(find(fNEST<-0.1));
x_NEST = zeros(N,1);
x_NEST(find(fNEST>0.1)) = 1;
x_NEST(find(fNEST<-0.1)) = -1;
plot(x); hold on; plot(x_NEST)
length(intersect(find(x==1),find(x_NEST==1)))+length(intersect(find(x==-1),find(x_NEST==-1)))
infoNEST.mse(end)  % 0.0108







%% (2.1) Subsampled Welsh-Hadamard transform A + ISTA
S = eye(N); S = S(datasample(1:N, K, 'Replace', false),:);
y = S*fwht(x);
y = y + sigma*max(abs(y))*randn(K,1);
A = @(x) S*fwht(x); At = @(x) ifwht(S'*x); AtA = @(x) At(A(x));
F.f = @(x,y) 0.5*sum((A(x) - y).^2); 
F.df = @(x,y) At(A(x) - y); 
G.f = @(x) norm(x,1);
G.prox = @(x, alpha) softThresh(x, alpha); 
L = 1;
obj.f = @(x,lambda,y) F.f(x,y) + lambda*G.f(x); % Regularized objective function
obj.L = L; % Lipschitz constant for the gradient
% ISTA
x0 = At(y);  % min L2-norm (energy) solution
lambda = 0.1*max(abs(x0));
x0 = zeros(N,1); %lambda = 0.01;
[fISTA, infoISTA.obj, infoISTA.mse,infoISTA.stopC] = ista(y, F, G, lambda, x0, 1, 1e-5, 200, x, @(x) x);
MSEista2 = infoISTA.mse;

infoISTA.obj
length(find(fISTA>0.1));
length(find(fISTA<-0.1));
x_ISTA = zeros(N,1);
x_ISTA(find(fISTA>0.1)) = 1;
x_ISTA(find(fISTA<-0.1)) = -1;
plot(x); hold on; plot(x_ISTA)
length(intersect(find(x==1),find(x_ISTA==1)))+length(intersect(find(x==-1),find(x_ISTA==-1)))
% 80 of the 100 spikes reconstructed
infoISTA.mse(end);  % 0.0108








%% (2.2) Subsampled Welsh-Hadamard transform A + FISTA
x0 = zeros(N,1);
[fNEST, infoNEST.obj, infoNEST.mse] = fista(y, F, G, lambda, x0, 1, 1e-5, 100, x, @(x) x);
MSEfista2 = infoNEST.mse;
infoNEST.obj
length(find(fNEST>0.1));
length(find(fNEST<-0.1));
x_NEST = zeros(N,1);
x_NEST(find(fNEST>0.1)) = 1;
x_NEST(find(fNEST<-0.1)) = -1;
plot(x); hold on; plot(x_NEST)
length(intersect(find(x==1),find(x_NEST==1)))+length(intersect(find(x==-1),find(x_NEST==-1)))
infoNEST.mse(end);  % 0.0106











%% (3.1) Gaussian random matrix A + ADMM
sigma = 0.005; y = A_matrix*x; y = y + sigma*max(abs(y))*randn(K,1);
A = @(x) A_matrix * x;
At = @(x) A_matrix' * x;
AtA = @(x) At(A(x));
% Gradient operators and the adjoint
%Dx = @(x)  [diff(x,1,2),zeros(size(x,1),1)];
%Dy = @(x)  [diff(x,1,1);zeros(1,size(x,2))];
%DxT = @(x) [-x(:,1),-diff(x(:,1:end-1),1,2),x(:,end-1)];
%DyT = @(x) [-x(1,:);-diff(x(1:end-1,:),1,1);x(end-1,:)];

% Initialisation
%E = {@(x) Dx(x); @(x) Dy(x)};
%Etr = {@(x) DxT(x); @(x) DyT(x)};
E = @(x) x;
Etr = @(x) x';

F = @(x) -x;
b = zeros(size(x0));

invLS = [];

% BEST PARAMETERS
para.stopTolerance = 1e-6;
para.maxIter = 1000;
para.rho = 1;
para.mu = 10; para.overRelaxPara = 1.5;

% ADMM with strucutred solve for || A u - y ||_2^2 + rho || u - q ||_2^2.
% To use generic iterative solver replace invLS with []
% shrinkage operator
x0 = At(y); lambda = 0.1*max(abs(x0));
Proxy = @(x, rho) softThresh(x, lambda*rho); 
[fADMM, v, w, iter, stopValue, uvIterates, info] = ADMM(y, A, At, invLS, E, Etr, F, b, Proxy, 0, para, x);
histogram(fADMM)
MSEadmm = info.mse;
figure

length(find(fADMM>0.1));
length(find(fADMM<-0.1));
x_ADMM = zeros(N,1);
x_ADMM(find(fADMM>0.1)) = 1;
x_ADMM(find(fADMM<-0.1)) = -1;
plot(x); hold on; plot(x_ADMM)
length(intersect(find(x==1),find(x_ADMM==1)))+length(intersect(find(x==-1),find(x_ADMM==-1)))





% check the convergence
% figure;
% semilogy(info.stopT);
% title('stopping values vs. iterations')





%% (3.2) Subsampled Welsh-Hadamard transform A + ADMM
S = eye(N); S = S(datasample(1:N, K, 'Replace', false),:);
y = S*fwht(x);
y = y + sigma*max(abs(y))*randn(K,1);
A = @(x) S*fwht(x); At = @(x) ifwht(S'*x); AtA = @(x) At(A(x));
[fADMM, v, w, iter, stopValue, uvIterates, info] = ADMM(y, A, At, invLS, E, Etr, F, b, Proxy, 0, para, x);
MSEadmm2 = info.mse;

histogram(fADMM)
length(find(fADMM>0.1));
length(find(fADMM<-0.1));
x_ADMM = zeros(N,1);
x_ADMM(find(fADMM>0.1)) = 1;
x_ADMM(find(fADMM<-0.1)) = -1;
plot(x); hold on; plot(x_ADMM)
length(intersect(find(x==1),find(x_ADMM==1)))+length(intersect(find(x==-1),find(x_ADMM==-1)))















plot(MSEista,'LineWidth',2); hold on; plot(MSEfista,'LineWidth',2); hold on; plot(MSEadmm,'LineWidth',2);
xlabel('Iteration');  ylabel('MSE')
legend('ISTA','FISTA','ADMM'); title('Random Gaussian Measurement Type')

plot(MSEista2,'LineWidth',2); hold on; plot(MSEfista2,'LineWidth',2); hold on; plot(MSEadmm2,'LineWidth',2);
xlabel('Iteration');  ylabel('MSE')
legend('ISTA','FISTA','ADMM');title('Welsh-Hadamard Measurement Type')


