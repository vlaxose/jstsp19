function A = UniformSphericalMatrix(n,p)
% UniformSphericalMatrix: creates random (Gaussian) spherical matrix
% Usage
%	A = UniformSphericalMatrix(n,p)
% Input
%	n           number of columns
%	p           number of rows
%                       
% Outputs
%	 A          random spherical matrix
%
A = randn(n,p);
for j = 1:p
    A(:,j) = A(:,j) ./ norm(A(:,j)); %orthogonalize columns
end
%
% Part of SparseLab Version:100
% Created Tuesday March 28, 2006
% This is Copyrighted Material
% For Copying permissions see COPYING.m
% Comments? e-mail sparselab@stanford.edu
%
