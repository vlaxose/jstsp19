function A = RandomSphericalMatrix(n,p)
% RandomSphericalMatrix: creates random (Gaussian) spherical matrix
% Usage
%	A = RandomSphericalMatrix(n,p)
% Input
%	n           number of columns
%	p           number of rows
%                       
% Outputs
%	 A          random spherical matrix
%
A = sign(randn(d,n) - .5);
z = find(A == 0);
A(z) = ones(size(z));
for j = 1:n
    A(:,j) = A(:,j) ./ norm(A(:,j));
end
%
% Part of SparseLab Version:100
% Created Tuesday March 28, 2006
% This is Copyrighted Material
% For Copying permissions see COPYING.m
% Comments? e-mail sparselab@stanford.edu
%
