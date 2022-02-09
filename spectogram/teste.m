clc; clear; close all;
l = 240;
a = 144;
im = zeros(a, l, 3);

for i = 1:a
im(i, :, 3) = [1*i]*255/a;
end
im = uint8(im);
imshow(im)
imwrite(im, "gradi.jpg");