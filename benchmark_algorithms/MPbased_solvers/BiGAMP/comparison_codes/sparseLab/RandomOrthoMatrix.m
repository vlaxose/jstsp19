function A = RandomOrthoMatrix(n,p)
% RandomOrthoMatrix: creates random orthogonal matrix
% Usage
%	A = RandomOrthoMatrix(n,p)
% Input
%	n           number of columns
%	p           number of rows
%                       
% Outputs
%	 A          random orthogonal matrix
%
r = n/d;
A = [];
for j = 1:r
    [Q,R] = qr(rand(d));
    A = [A,Q];
end
%
% Part of SparseLab Version:100
% Created Tuesday March 28, 2006
% This is Copyrighted Material
% For Copying permissions see COPYING.m
% Comments? e-mail sparselab@stanford.edu
%
