% Run learner solution.
F.f = @(x) x(1).^2 + x(2).^2;
F.df = @(x) [2*x(1); 2*x(2)];

% Point x0 = [1; 3]
x0 = [1; 3];
p0 = [-2; -6];
alpha0 = 1;
opts.rho = 0.1;
opts.c1 = 1e-4;
[alpha, info] = backtracking(F, x0, p0, alpha0, opts);