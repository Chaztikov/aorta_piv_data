function res = cltheat2(t,y,yp)

    n=length(y);

    A = randn(n);
    res = cltheat(t,y,yp,A)
end