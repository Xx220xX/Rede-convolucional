
function [S_r, f_r, t_r] = mspc(x, n = min(256, length(x)), Fs = 2, window = hanning(n), overlap = ceil(length(window)/2))

  if (nargin < 1 || nargin > 5)
    print_usage ();
  endif

  if (! isnumeric (x) || ! isvector (x))
    error ("specgram: X must be a numeric vector");
  endif

  x = x(:);

  ## if only the window length is given, generate hanning window
  if (isscalar (window))
    window = hanning (window);
  endif

  ## should be extended to accept a vector of frequencies at which to
  ## evaluate the Fourier transform (via filterbank or chirp
  ## z-transform)
  if (! isscalar (n))
    error ("specgram: N must be a scalar, vector of frequencies not supported");
  endif

  if (length (x) <= length (window))
    error ("specgram: segment length must be less than the size of X");
  endif

  ## compute window offsets
  win_size = length(window);
  if (win_size > n)
    n = win_size;
    warning ("specgram fft size adjusted to %d", n);
  endif
  step = win_size - overlap;

  ## build matrix of windowed data slices
  offset = [ 1 : step : length(x)-win_size ];
  S = zeros (n, length(offset));
  for i=1:length(offset)
    S(1:win_size, i) = x(offset(i):offset(i)+win_size-1) .* window;
  endfor

  ## compute Fourier transform
  S = fft (S);

  ## extract the positive frequency components
  if rem(n,2)==1
    ret_n = (n+1)/2;
  else
    ret_n = n/2;
  endif
  S = S(1:ret_n, :);

  f = [0:ret_n-1]*Fs/n;
  t = offset/Fs;
  if nargout==0
    imagesc(t, f, 20*log10(abs(S)));
    set (gca (), "ydir", "normal");
    xlabel ("Time")
    ylabel ("Frequency")
  endif
  if nargout>0, S_r = S; endif
  if nargout>1, f_r = f; endif
  if nargout>2, t_r = t; endif

endfunction
