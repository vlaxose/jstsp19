function Mat = mat2rgb(Mat,imsize)

% convert matrix to movie
%
% Mat -- m x n matrix with
%        m = imsize(1) * imsize(2)
%        n = number of frames * 3
%
% Yin Zhang 2009

[m,n] = size(Mat);
if mod(n,3) ~= 0;
    error('number of columns must be divisible by 3');
end
if m ~= prod(imsize)
    error('number of pixels mismatches image size');
end

% translate Mat to [0,1] in double precision 
Mat = single(Mat);
minel = min(Mat(:));
maxel = max(Mat(:));
Mat = (Mat - minel) / (maxel - minel);
Mat = min(1,max(0,Mat));
Mat = reshape(Mat,[imsize(1) imsize(2) 3 n/3]);

