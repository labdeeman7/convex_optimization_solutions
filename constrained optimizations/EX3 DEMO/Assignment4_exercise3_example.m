%% *COMPGV19: Assignment 4 Exercise 3 DEMO
%
% Bolin Pan, Marta Betcke, Kiko Rullan
% 
% This is an example for Assignment 4 Exercise 3 DEMO to demonstrate how to
% use ISTA, FISTA and ADMM with total variation penalty on images.
%
% The prox of TV is solved by the Primal Dual Hybrid Gradient method given 
% by Goldstein, Tom and Li, Min and Yuan, Xiaoming
% "Adaptive primal-dual splitting methods for statistical learning and image
% processing", Advances in Neural Information Processing Systems,2089-2097,
% 2015
% 
% This script requires the PDHG toolbox which can be downloaded from
% https://www.cs.umd.edu/~tomg/projects/pdhg/
%
% The ADMM_TV function is used to solve the total variation denoising
% problem only.

%==================================================
% SETUP
%==================================================
% Set path to ISTA
addpath ../ISTA 
addpath PDHG
addpath adaptive_pdhg/solvers

clear all; 
close all;
% Set states for repeatable experiments
rng('default');


%%

%==================================================
% IMAGE DEFINITION
%==================================================

% Read image and rescale
f = imread('Cameraman256.png');
f = double(f);
f = f/max(f(:));

% Get the size of the image in pixels
[nx,ny] = size(f);
% Construct parameters structure
parameters = [];
parameters.nx = nx;
parameters.ny = ny;

% Construct Gaussian blurring kernel with standard deviation 1.5 (in units
% of pixel length) and a size of 5 standard deviations
sigmaA = 1.5;
sizeKernel = ceil(sigmaA * [5,5]);
sizeKernel = sizeKernel + mod(sizeKernel+1,2);
Aker = fspecial('gaussian',sizeKernel,sigmaA);
% assign to parameters
parameters.Aker = Aker;


%%

%==================================================
% OPERATOR DEFINITION: Blurring with Gaussian kernel
%==================================================
A = @(x) blur(x,parameters); 
At = A; % At = A for Gaussian convolution 
%%% I dont understand this.
AtA = @(x) At(A(x));


%%

%==================================================
% GENERATE NOISY MEASUREMENTS
%==================================================
% Blurr image
gClean = blur(f,parameters);
gClean = reshape(gClean,nx,ny);
% Add Gaussian noise with standard deviation of 0.05
sigmaNoise = 0.05;
gNoisy = gClean + sigmaNoise * randn(size(gClean));

% Display the imageS
figure;
subplot(1,2,1); imagesc(reshape(f,nx,ny)); title('f'); axis image; colormap gray; colorbar
subplot(1,2,2); imagesc(reshape(gNoisy,nx,ny)); title('gNoisy'); axis image; colormap gray; colorbar


%%

%==========================================================================================
%========================                                        ==========================
%========================          SOLVERS (comment in/out)       ==========================
%========================                                        ==========================
%==========================================================================================


%%

%==================================================
% Initialization ISTA/FISTA ADMM
%==================================================
% Initial guess: 
x0 = At(gNoisy);  % min L2-norm (energy) solution
%x0 = zeros(N,1); % all 0s

% Regularization parameter
lambda = 0.01*max(abs(At(gNoisy)));


%%

%==================================================
% FUNCTION DEFINITION ISTA/FISTA
%==================================================
% L2 Data fit term 
F.f = @(x,y) 0.5*sum((A(x) - y).^2); 
% Gradient of the data fit term
F.df = @(x,y) At(A(x) - y); 
  
% TV regularization term
G.f = @(x) TVnorm(x,parameters);
% Proximal operator wrt TV-norm TV(x)
G.prox = @(x, alpha) reshape(pdhg_tv(reshape(x,nx,ny), 1/alpha),[],1); 

% Compute the Liptchitz constant of AtA
tol = 1e-3; % tolerarance 
vInit = rand(nx*ny,1);
[L, vec] = powerIteration(vInit,AtA,tol);

% Objective function as in eg gradient descent 
obj.f = @(x,lambda,y) F.f(x,y) + lambda*G.f(x); % Regularized objective function
obj.L = L; % Lipschitz constant for the gradient


%%

%==================== 
% ISTA 
%==================== 
[fISTA, infoISTA.obj, infoISTA.mse,infoISTA.stopC] = ista(gNoisy(:), F, G, lambda, x0, 1, 5e-4, 100, f(:), At);
figure;
subplot(1,2,1); imagesc(reshape(fISTA,nx,ny)); title('ISTA'); axis image; colormap gray; colorbar
subplot(1,2,2); imagesc(reshape(fISTA,nx,ny) - f); title('Error: ISTA'); axis image; colormap gray; colorbar


%%

%==================== 
% FISTA: ISTA + Nesterov acceleration
%==================== 
[fNEST, infoNEST.obj, infoNEST.mse] = fista(gNoisy(:), F, G, lambda, x0, 1, 5e-4, 100, f(:), At);
figure;
subplot(1,2,1); imagesc(reshape(fNEST,nx,ny)); title('FISTA'); axis image; colormap gray; colorbar
subplot(1,2,2); imagesc(reshape(fNEST,nx,ny) - f); title('Error: FISTA'); axis image; colormap gray; colorbar


%% compare convergence ISTA vs FISTA

% objective function
figure, 
plot(infoISTA.obj, 'LineWidth', 2); hold on
plot(infoNEST.obj, 'LineWidth', 2); 
title('Relative variation of the regularized objective function');
xlabel('k'); 
legend('ISTA', 'FISTA')
grid on;

% mse error
figure, 
plot(infoISTA.mse, 'LineWidth', 2); hold on
plot(infoNEST.mse, 'LineWidth', 2); 
title('MSE, ISTA vs. FISTA');
xlabel('k'); 
legend('ISTA', 'FISTA')
grid on;


%%

%==================================================
% SOLVE USING ADMM
%==================================================
% Gradient operators and the adjoint
Dx = @(x)  [diff(x,1,2),zeros(size(x,1),1)];
Dy = @(x)  [diff(x,1,1);zeros(1,size(x,2))];
DxT = @(x) [-x(:,1),-diff(x(:,1:end-1),1,2),x(:,end-1)];
DyT = @(x) [-x(1,:);-diff(x(1:end-1,:),1,1);x(end-1,:)];

% Initialisation
E = {@(x) Dx(x); @(x) Dy(x)};
Etr = {@(x) DxT(x); @(x) DyT(x)};
F = @(v) -v;
b = 0*reshape(x0,nx,ny);

% shrinkage operator
Proxy = @(x, rho) softThresh(x, lambda*rho); 

invLS = []; % blurring operator


para.stopTolerance = 1e-6;
para.maxIter = 200;
para.rho = 1;
para.mu = 5;

% ADMM with strucutred solve for || A u - y ||_2^2 + rho || u - q ||_2^2.
% To use generic iterative solver replace invLS with []
[fADMM, v, w, iter, stopValue, uvIterates, info] = ADMM_TV(gNoisy, A, At, invLS, E, Etr, F, b, Proxy, reshape(x0,nx,ny), para);

% Plot results
figure;
subplot(1,2,1); imagesc(reshape(fADMM,nx,ny)); title('ADMM'); axis image; colormap gray; colorbar
subplot(1,2,2); imagesc(reshape(fADMM,nx,ny) - f); title('Error: ADMM'); axis image; colormap gray; colorbar

% check the convergence
figure;
semilogy(info.stopT);
title('stopping values vs. iterations')


%%

%==================================================
% FUNCTION HANDLES
%==================================================

% 2D Gaussian convolution
function BlurredIm = blur(im,parameters)
% imblur is the forward convolution operator for computing g = Af
%  g = imblur(f,parameters)
%
%  INPUT:
%   im - a Nx x Ny image
%   parameters:
%     - Aker - blurring kernel
%
%  OUTPUTS:
%   BlurredIm - the blurred image

Aker = parameters.Aker;
im = reshape(im,parameters.nx,parameters.ny);
% Perform the forward convolution
BlurredIm  = imfilter(im,Aker,'circular');
BlurredIm  = BlurredIm(:);

end

% Power method to compute the Lipschitz constant
function [L, vReturn] = powerIteration(v,A,tol)
% Calculates the largest eigenvalue (Lipschitz constant) and corresponding 
% eigenvector of operator A by the power method using v as the starting vector 

% Initialization
L = inf;
Lold = 1;

% Loop to compute the biggest eigenvalue of A
while abs(L - Lold) > tol
    Lold = L;
    vnew = A(v);
    % Eigenvalues
    L = norm(vnew)/norm(v);
    % Normalization
    v = vnew;
end

vReturn = v/norm(v);
end

% compute the TV norm
function TVenergy = TVnorm(x,parameters)

% Define discrete derivatives in x and y direction
Dx = @(x)  [diff(x,1,2),zeros(size(x,1),1)];
Dy = @(x)  [diff(x,1,1);zeros(1,size(x,2))];

% Compute the TV norm
x = reshape(x,parameters.nx,parameters.ny);
Dx = Dx(x);
Dy = Dy(x);
absDxDy = abs(Dx) + abs(Dy);
TVenergy = sum(absDxDy(:));
end

