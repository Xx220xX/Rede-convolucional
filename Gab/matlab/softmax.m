clc;clear all;close all;
n = 30;

function y = sf(x)
  expo = exp(x);
  y = expo / sum(expo);
end
erro = 0;
jacobiano = @(y)y.*(1-y) .* eye(length(y)) + (1-eye(length(y))).*(-y'*y);
for k=1:1e5
x = randn(n,1);
y = sf(x);
yn = sf(x - max(x));
erro = erro + sum((jacobiano(y) *ones(n,1) - jacobiano(yn)*ones(n,1)).^2);
end
0.5*erro/k