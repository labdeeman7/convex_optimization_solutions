% For computation define as function of 1 vector variable
F.f = @(x) 100.*(x(2) - x(1)^2).^2 + (1 - x(1)).^2; 
F.df = @(x) [-400*(x(2) - x(1)^2)*x(1) - 2*(1 - x(1)); 
              200*(x(2) - x(1)^2)];  
F.d2f = @(x) [-400*(x(2) - 3*x(1)^2) + 2, -400*x(1); -400*x(1), 200]; 

% Point
x0 = [1.2; 1.2];

% Delta 
delta = 0.1;

p = solverCMdogleg(F, x0, delta);
disp(p);