function y = dfalan(x)
  y = x;
  for i = 1:size(x,1)
    for j = 1:size(x,2)
      for k = 1:size(x,3) 
        if x(i,j,k)>1
          y(i,j,k) = 0.419978/x(i,j,k);
        elseif x(i,j,k) < -1
          y(i,j,k) = -0.419978/x(i,j,k);
        else 
          y(i,j,k) = 1 - tanh(x(i,j,k)) .^2;
        end
      end
    end
  end
end
