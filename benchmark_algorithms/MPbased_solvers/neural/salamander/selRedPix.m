function Imax = selRedPix(linWt, dist, xpix, ypix)
% selRedPix:  Given a linear weight estimate, the routine selects a subset
% of pixel, Imax, in a square of (d+1) x (d+1) around the pixel with the
% maximal response.

% Find the pixel with the maximum sum weight
[mm,im] = max(sum(abs(linWt)));
imaxy = floor( im/xpix );
imaxx = im - imaxy*xpix;

% Find subregion to search on around the pixels
dx = (abs((1:xpix) - imaxx) <= dist)';
dy = (abs((1:ypix) - imaxy) <= dist);
Isq = repmat(dx,1,ypix) & repmat(dy,xpix,1);
Ilin = Isq(:);
Imax = find(Ilin);