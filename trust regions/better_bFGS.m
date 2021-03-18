%L-BFGS method, algorithm 7.4 and 7.5 from norcedal and wright second edition

m = 5; %pick any number between 3 and 20 should work
k = 1; %count iterations, matlab indexing starts from 1
n = length(x_k); 
%
%

s_k = x_k-x_k_1;
y_k = F.df(x_k) - F.df(x_k_1);
%
%
S = zeros(n,m); % store m values of s_k
Y = zeros(n,m); % store m values of y_k
%
%

H0 = (s_k'*y_k)/(y_k'*y_k)*eye(n); %initial H
p_k = zeros(length(F.df(x_k)),1); %initialize descent direction 

if (k<=m)
    % update S,Y
    S(:,k) = s_k;
    Y(:,k) = y_k;
    p_k = -1 * lbfgs(g1,S(:,1:k),Y(:,1:k),H0); 
else
    S(:,1:(m-1))=S(:,2:m); % shift to the left
    Y(:,1:(m-1))=Y(:,2:m);
    S(:,m) = s_k; % at the new ones
    Y(:,m) = y_k;    
    p_k = -1 * lbfgs(g1,S,Y,H0);
end  

 alpha_k = ls(x_k, p_k, alpha0);

% Update x_k and f_k
x_k_1 = x_k;
x_k = x_k + alpha_k*p_k;

%%%
%%%
    
function H_new = lbfgs(grad,S,Y,H0)
% Returns our approximated Hessian  multiplied by gradient, H*g
% Input
%   S:    Memory of s_i (n by k) 
%   Y:    Memory of y_i (n by k) 
%   g:    gradient (n by 1)
%   H0 : Initial hessian
% Output
%   Hg    the the approximate inverse Hessian multiplied by the gradient g


    [n,k] = size(S);
    for i = 1:k
        rho(i,1) = 1/(Y(:,i)'*S(:,i));
    end

    q = zeros(n,k+1);
    alpha =zeros(k,1);
    beta =zeros(k,1);

    % step 1
    q(:,k+1) = grad;

    % loop 1
    for i = k:-1:1
        alpha(i) = rho(i)*S(:,i)'*q(:,i+1);
        q(:,i) = q(:,i+1)-alpha(i)*Y(:,i);
    end

    % Multiply by Initial Hessian
    r = H0*q(:,1);

    % loop 2
    for i = 1:k
        beta(i) = rho(i)*Y(:,i)'*r;
        r = r + S(:,i)*(alpha(i)-beta(i));
    end
    H_new=r; %approximate hessian
end 

