syms x y a
f = 2*x + 4*y + x^2 - 2*y^2;

figure(1)
fcontour(f, [-50,50])
figure(2)
fsurf(f, [-50,50])
xlabel('x label')
ylabel('y label')
zlabel('z label')


