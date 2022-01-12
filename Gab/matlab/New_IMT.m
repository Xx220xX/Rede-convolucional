function New_IMT()
index = 1
w = rand(3,3);
figure(index); 
clf
ylim([0,size(w,1)+1]);
xlim([0,size(w,2)+1]);
for i = 1:size(w,1)
  for j = 1:size(w,2)
    text(i,j,sprintf("%f",w(j,i)),'color',[1,0,0]);
  end
end
  