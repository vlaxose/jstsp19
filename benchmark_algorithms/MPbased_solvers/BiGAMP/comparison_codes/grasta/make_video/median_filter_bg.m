function [ fg, bg, bgbuf, k] = median_filter_bg( img, bgbuf, N, k  )
%MEDIAN_FILTER_BG Summary of this function goes here
%   Detailed explanation goes here

if k<N,
   if k>1,
       bg = median(bgbuf,2);

   else
       bg = zeros(size(img));
   end
else
   bg = median(bgbuf,2);
end

idx = mod(k,N);
if idx==0,
    idx = N;
end
bgbuf(:,idx) = img;

k = k+1;

fg = img - bg;


end

