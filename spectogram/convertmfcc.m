function convertmfcc(colmap,prefixe_dir,out,itens)
  mkdir(out);
  base = prefixe_dir;
  map = colmap;
  h = waitbar(0,['trabalhando em ' base ' -> ' out]);
   
    n_frequencias = 256;
    window = hanning(n_frequencias);
    plot(window)
  for i = 1:itens
    nome_arquivo_in = sprintf('%s (%d).wav',base,i);
    nome_arquivo_out = sprintf('%s (%d).hex',out,i);
   
    [y,fs] = audioread(nome_arquivo_in);
    frequencia_amostragem = fs;
    
  
    sinal = y;
    
    [S,f,t] = mspc(sinal,n_frequencias,frequencia_amostragem,window);
    im =  20*log10(abs(S));
    im = im(end:-1:1,:);
    
    
    minv = min(im(:));
    maxv = max(im(:));
    ncol = size(map,1);
    s = round(1+(ncol-1)*(im-minv)/(maxv-minv));
    
    im = ind2rgb(s,map);
    im = uint8(im*255);
    file = fopen(nome_arquivo_out,'wb');
    fwrite(file,uint32(size(im,1)));
    fwrite(file,uint32(size(im,2)));
    fwrite(file,uint32(size(im,3)));
    for z = 1:3
      for x=1:size(im,1)
        fwrite(file,im(x,:,z));
      end
    end
    fclose(file);
    %imshow(im)
    waitbar(i/itens,h);
  end
  close(h);
  
  end