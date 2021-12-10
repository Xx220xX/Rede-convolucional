function img(data)
	data = data(:);
	data(data==nan)=0;
	mx = max(data);
	mn = min(data);
	mx = mx - mn;
	if mx == 0
		mx = 1;
	end
	data = (data -mn)/mx;
	imshow(data);s
end