# Teoria

Seja x a entrada da camada de batch normalization

A saída é espressa por:

<img style="background-color: #fff"  href="#" src="https://latex.codecogs.com/svg.latex?&space;y_i=f(x)=\frac{x_i-m(x_i)}{s(x_i)}\gamma+\beta" alt="equacao"/>

As funções <img style="background-color: #fff"  href="#" src="https://latex.codecogs.com/svg.latex?&space;m(x_i)" alt=""/> e
<img style="background-color: #fff"  href="#" src="https://latex.codecogs.com/svg.latex?&space;s(x_i)" alt=""/> são definidas como:


<img style="background-color: #fff"  href="#" src="https://latex.codecogs.com/svg.latex?&space;m(x_i)=\frac{\sum_{j=0}^{M} x_j}{M}" alt=""/>

- 

<img style="background-color: #fff"  href="#" src="https://latex.codecogs.com/svg.latex?&space;s(x_i)=\sqrt{\frac{\sum_{j=0}^{M}(x_j-m(x_i))^2}{M}+\epsilon}" alt=""/>


A derivada de da saida em relacao a entrada é:

<img style="background-color: #fff"  href="#" src="https://latex.codecogs.com/svg.latex?&space;f'(x_i)=\frac{s(x_i)-x_is'(x_i)-\frac{s(x_i)}{M}+m(x_i)s'(x_i)}{s(x)^2}" alt=""/>

simplificando:

<img style="background-color: #fff"  href="#" src="https://latex.codecogs.com/svg.latex?&space;f'(x_i)=\frac{s(x_i)\cdot\frac{M-1}{M}+(m(x_i)-x_i)\cdot%20s'(x_i)}{s(x_i)^2}" alt=""/>

e

 <img style="background-color: #fff"  href="#" width="80%" src="https://latex.codecogs.com/svg.latex?&space;s'(x_i)=\frac{-1}{s(x_i)M^2}\cdot((x_i-m(x_i))(M-1)+\sum_{j\neq%20i}(x_j-m(x_i )))" alt=""/>



- As variaveis que serao armazenas
    - media = m(x) = sum(x)/length(x)
    - diferenca =  x - media
    - somaDiferenca = sum(diferenca) 
    - variancia = s(x) = srqt(sum(diferenca^2)/m+epislon)
    - gradvariancia = -1/(variancia* m^2) * ((diferenca*(m-1) + direnca-x+media) 
    - dz = f'(x)*gradNxt(x) = gradNext(x) * (variancia * (m-1)/m +   gradvariancia)/(variancia^2)                                                  
