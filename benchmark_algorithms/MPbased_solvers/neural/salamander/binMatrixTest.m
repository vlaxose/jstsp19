clear all;
clear classes;

if 0
nrow = 100000;
ncol = 100;
ndly = 30;

B = BinaryMatrix(nrow,ncol);
H = randn(ndly,ncol);
disp('fir');
tic
y = B.firFilt(H);
toc
disp('fir tr');
tic
H1 = B.firFiltTr(y,ndly);
toc
disp('done');

end

nrow = 10;
ncol = 60;
ndly = 4;
A = (rand(nrow,ncol) < 0.5);
H = randn(ndly,ncol);

B = BinaryMatrix(nrow, ncol);
for irow = 1:nrow
    B.setRow(irow, A(irow,:));
end

% Create Toeplitz matrix
n = ncol*ndly;
T = zeros(nrow,n);
for icol = 1:ncol
    for idly = 1:ndly
        T(idly:nrow,(icol-1)*ndly+idly) = A(1:nrow-idly+1,icol);
    end
end


% Filter test
y = B.firFilt(H);
h = H(:);
y1 = T*h;

% Filter Transpose test
y2 = zeros(nrow+ndly-1,1);
for icol = 1:ncol
    y2 = y2 + conv(double(A(:,icol)), H(:,icol) );
end

% Filter transpose
H1 = B.firFiltTr(y,ndly);
h = T'*y;
H2 = reshape(h,ndly,ncol);



