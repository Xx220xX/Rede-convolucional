clc
close all
clear all

ty = 3;
tx = 4;
tz = 5;
x = 2
y = 2
z = 2
l = 1
k = l*(tx*ty*tz) + z*(tx*ty) + x*ty + y

y_ = mod(k,ty)
x_ = mod(k - y,ty*tx)/ty
z_ = mod(k- x_*ty - y_,tx*ty*tz)/(ty*tx)
l_ = (k -z_*tx*ty -x_*ty - y_)/(tx*ty*tz)