clc;
clear all;
close all
fs = 44100;
recObj = audiorecorder(fs,24,1);
disp('Start speaking.')
recordblocking(recObj, 3);
disp('End of Recording.');
play(recObj);
y = getaudiodata(recObj);

pt = y.^2;
mx = max(pt)/100;
for j =1:length(y)
  if pt(j) >= mx
    i = j;
    break
  end
end
for j =length(y):-1:1
  if pt(j) >= mx
    j = j;
    break
  end
end
i
j
c = y(i:j);
sound(y,fs)
plot(y)
figure
plot(c)
figure
subplot(2,1,1)
specgram(y)
subplot(2,1,2)
specgram(c)
%Y = fft(y)/length(y);
%Y = fftshift(Y);
%plot(abs(Y))