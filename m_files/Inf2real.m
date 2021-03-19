function y = Inf2real(x)
x(x(:) == -Inf) = realmin;
x(x(:) == Inf) = realmax;
y = x;
end