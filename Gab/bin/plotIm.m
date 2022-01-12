function plotIm(index,w)
  figure(index);
  clf;
  ylim([-1,size(w,1)+1]);
  xlim([-1,size(w,2)+1]);
  for i = 1:size(w,1)
    for j = 1:size(w,2)
      text(j,size(w,1)-i,sprintf("%f",w(i,j)));
    end
  end
end