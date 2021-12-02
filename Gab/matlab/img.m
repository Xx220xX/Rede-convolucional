function img(data)
  mx = max(data(:));
  mn = min(data(:));
  mx = mx - mn;
  if mx == 0
    mx = 1;
  end
  data = (data -mn)/mx;
  imshow(data);s
end