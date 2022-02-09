function b = mshape(w,shape)
  if length(shape) == 2
    
    b = reshape(w,[shape(2),shape(1)])';
  elseif length(shape) == 3
    b = zeros(shape);
    for i = 1:shape(1)
      for j = 1:shape(2)
        for z = 1:shape(3)
          b(i,j,z) = w((z-1)*shape(2)*shape(1)+(i-1)*shape(2)+j);
        end
      end
    end
  elseif length(shape) == 4
    b = zeros(shape);
    for l = 1:shape(4)
      for i = 1:shape(1)
        for j = 1:shape(2)
          for z = 1:shape(3)
            b(i,j,z,l) = w((l-1)*shape(3)*shape(2)*shape(1)+(z-1)*shape(2)*shape(1)+(i-1)*shape(2)+j);
          end
        end
      end
    end
  end
end
