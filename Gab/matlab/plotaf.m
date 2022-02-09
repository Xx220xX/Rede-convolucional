clc;close all;clear all;
x = -20:0.1:20;
hold on;
plot(x,alan(x),'DisplayName','alan')
plot(x,tanh(x),'DisplayName','tanh')
title('Ativação')
grid on
xlabel('entrada')
ylabel('saída')
legend

figure
hold on;
plot(x,dfalan(x),'DisplayName','alan')
plot(x,1-tanh(x).^2,'DisplayName','tanh')
title('Derivada')
grid on
xlabel('entrada')
ylabel('saída')
legend
saveas(1,'ativa.png')
saveas(2,'deriva.png')