clc
clear all
close all

function da=dma(ds,filtro,entrada,p)
  da = entrada*0;
  M = size(filtro)(1);
  N = size(filtro)(2);
  
  for i=1:size(da)(1)
    for j=1:size(da)(2)
      for z=1:size(da)(3)
        min_m = 1;
        if size(da)(1)-i-M<0 
          min_m =  -size(da)(1)+i+M;
        end
        max_m = M;
        if i-M<0 
          max_m =  i;
        end
        min_n = 1;
        if size(da)(2)-j-N<0 
          min_n =  -size(da)(2)+j+N;
        end
        max_n = N;
        if j-N<0 
          max_n =  j;
        end
        
        for m=min_m:max_m
          for n=min_n:max_n
            a = (i - m)/p+ 1;
            b = (j - n)/p+ 1;
            for l=1:size(ds)(3)
              
              da(i,j,z) = da(i,j,z) + filtro(m,n,z,l)*ds(a,b,l);
            end
          end
        end
      end
    end
  end
  
end
function gradfiltro =dmconv(ds,filtro,entrada,p)
  gradfiltro = filtro*0;
  for i=1:size(ds)(1)
    for j=1:size(ds)(2)    
      for z=1:size(filtro)(3)
        
        for m=1:size(filtro)(1)
          for n=1:size(filtro)(2)
            for l=1:size(filtro)(4)
              gradfiltro(m,n,z,l) = gradfiltro(m,n,z,l) + entrada(i*p+m-1, j*p+n-1,z)*ds(i,j,l);
            endfor     
          endfor
        endfor
      endfor   
    endfor    
  endfor      
end

function saida=mconv(filtro,entrada,p)
  sf = size(filtro);
  sin = size(entrada);
  dim =  sin(3);
  sin = sin(1:2);
  saida = zeros((sin(1:2) - sf(1:2))/p + 1);
  
  for i=1:size(saida)(1)
    for j=1:size(saida)(2)
      tmp = filtro .* entrada(i*p:i*p+sf(1)-1,j*p:j*p+sf(2)-1,:);;
      saida(i,j) =  sum(tmp(:));
    end
  end
end

function  testconv()
  entrada = zeros(3,3,3);
  l = 0;
  for k=1:3
    for i=1:3
      for j = 1:3
        entrada(i,j,k) = l/27;
        l = l+1; 
      end
    end 
  end
  filtros = zeros(2,2,3,2);
  l = 1;
  for k=1:3
    for i=1:2
      for j = 1:2
        filtros(i,j,k,1) = l/(l+23);
        filtros(i,j,k,2) = l/(l+11);
        l = l+1;
      end
    end
  end
  target = zeros(2,2,2);
  l = 1;
  for k=1:size(target)(3)
    for i=1:size(target)(1)
      for j = 1:size(target)(2)
        target(i,j,k) = l/49;
        l = l+1; 
      end
    end 
  end
  ep = [];
  for i=1:1
    
    
    saida  = zeros(2,2,2);
    saida(:,:,1) = mconv(filtros(:,:,:,1),entrada,1);
    saida(:,:,2) = mconv(filtros(:,:,:,2),entrada,1);
    
    erro = saida - target;
    %ep = [ep norm(erro(:))];
    
    gradfiltros = zeros(2,2,3,2);
    gradfiltros = dmconv(erro, filtros, entrada, 1);
    gradIn = dma(erro,filtros,entrada,1);
    filtros = filtros - 0.1*gradfiltros;
    gradIn
    ep = [ep norm(gradIn(:))];
  endfor
  plot(ep)
  min(ep)
end

testconv