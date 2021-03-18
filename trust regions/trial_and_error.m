a = @(x) x;
for c = 1:3
    if(c == 1)
        c
        a = @(x) x/2;
        z = a(3)
    else
        z = a(3)
    end
end
    