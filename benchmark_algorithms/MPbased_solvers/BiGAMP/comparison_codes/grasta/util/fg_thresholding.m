function [ fg ] = fg_thresholding( x, thresh)
%FG_THRESHOLDING Summary of this function goes here
%   Detailed explanation goes here

x = abs(x);
% 1. normalize abs(x) into [0,1]
x = x/max(max(x));

fg_idx = x>thresh;

fg = zeros(size(x));

fg(fg_idx) = 1;

end

