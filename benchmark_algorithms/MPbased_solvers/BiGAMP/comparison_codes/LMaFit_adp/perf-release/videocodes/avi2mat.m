function [Mat,imsize] = avi2mat(filename)

% convert an avi file to a matrix

warning off;
vid = aviread(filename);
nFrames = length(vid);
imsize = size(vid(1).cdata(:,:,1));
nPix = prod(imsize);
Mat = zeros(nPix,3*nFrames,'uint8');

for i = 1 : nFrames
    J = (1:3) + 3*(i-1);
    Mat(:,J) = reshape(vid(i).cdata,[nPix 3]);
end
