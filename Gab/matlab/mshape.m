function B = mshape(A,shape)
  if length(shape) == 2
    
    B = reshape(A,[shape(2),shape(1)])';
  elseif length(shape) == 3
    B = zeros(shape);
    for i = 1:shape(1)
      for j = 1:shape(2)
        for z = 1:shape(3)
          B(i,j,z) = A((z-1)*shape(2)*shape(1)+(i-1)*shape(2)+j);
        end
      end
    end
  elseif length(shape) == 4
    B = zeros(shape);
    for l = 1:shape(4)
      for i = 1:shape(1)
        for j = 1:shape(2)
          for z = 1:shape(3)
            B(i,j,z,l) = A((l-1)*shape(3)*shape(2)*shape(1)+(z-1)*shape(2)*shape(1)+(i-1)*shape(2)+j);
          end
        end
      end
    end
  end
end
