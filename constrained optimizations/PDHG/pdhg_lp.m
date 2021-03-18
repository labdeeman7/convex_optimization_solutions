% Solve linear program using adaptive PDHG.  The problem solved is
%            minimixe     <f,x>
%            subject to
%                     Ain*x<=bin     (inequalities)
%                     Aeq*x=beq      (equalities)
%                     lb <= x <= ub  (lower/upper bound)
%  Inputs...
%    f     : A column vector defining the objective function
%    Ain   : Matrix of inequality coefficients
%    bin   : Vector containing inequality rhs
%    Aeq   : Matrix of equality coefficients
%    beq   : Vector containing equality rhs
%    ub    : Vector containing upper bound values
%    lb    : Vector containing lower bound values
%    opts  : An optional argument for customizing the behavior of 
%              pdhg_adaptive
%  Outputs...
%    sol : The approximate solution
%    out : A struct containing convergence data generated by the
%                function 'pdhg_adaptive' when it solved the problem.
%   Note:  If no equalities are needed, one can choose Aeq=beq=[].
%   Similarly, one may choose Ain=bin=[].  However, one of these two types
%   of constraints must be a non-empty matrix.

%  This File requires that 'pdhg_adaptive.m' be in your current path. 
%
%  For an explanation of PDHG and how it applies to this problem, see
%    "Adaptive Primal-Dual Hybrid Gradient Methods for Saddle-Point
%    Problems"  available at <http://arxiv.org/abs/1305.0546>
%


function [ sol, out ] = pdhg_lp( f,Ain,bin,Aeq,beq, ub, lb, opts )

    %% Record problem dimensions
    Nvar = size(Ain,2);
    Nin = size(Ain,1);
    Neq = size(Aeq,1);

    %%  Create the linear operator to enforce constraints
    M = [Ain ; Aeq];
    b = [bin ; beq];
    %% Precondition the problem
    % Define left and right preconditioners
    LeftPrecond = (sum(abs(M),2)).^-.5;
    RightPrecond = ((sum(abs(M),1)).^-.5)';
    %LeftPrecond = LeftPrecond./LeftPrecond;
    %RightPrecond = RightPrecond./RightPrecond*2;
   
    %  Apply them to M
    M = diag(LeftPrecond)*M*diag(RightPrecond);
    b =  LeftPrecond.*b;
    bin = b(1:Nin);
    beq = b(Nin+1:end);
    f = f.*RightPrecond;
    %opts.L = .99;
    
    %% Define the ingredients PDHG needs to solve this problem
    fProx = @(x,tau) min(ub,max(x -tau*f  ,lb));
    gProx = @(y,sigma) [ max(y(1:Nin)-sigma*bin,0) ; y(Nin+1:end)-sigma*beq ];
    A = @(x) M*x;
    At = @(y) M'*y;
    %Initial Guess
    x0 = zeros(Nvar,1);
    y0 = zeros(Nin+Neq,1);
       
    
    %eps = .0001;
   
    %%  Some optional convergence parameters
    if ~exist('opts','var')
        opts = [];
    end
    opts.maxIters = 100000;
    
    [sol ,out]= pdhg_adaptive(x0,y0,A,At,fProx,gProx,opts); 
    
    sol = sol.*RightPrecond;
    

end

