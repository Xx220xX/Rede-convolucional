function y = alan(x)
  y = x;
  for i = 1:size(x,1)
    for j = 1:size(x,2)
      for k = 1:size(x,3) 
        if x(i,j,k)>1
          y(i,j,k) = log10(x(i,j,k)) + 0.7615941559557649;
        elseif x(i,j,k) < -1
          y(i,j,k) = -log10(-x(i,j,k)) - 0.7615941559557649;
        else 
          y(i,j,k) = tanh(x(i,j,k));
        end
      end
  end
end
