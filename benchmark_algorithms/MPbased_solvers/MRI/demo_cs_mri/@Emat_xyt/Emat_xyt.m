function  res = Emat_xyt(mask,b1)

%res = Emat_xyt(mask,b1)
%
%
%	Implementation of parallel MRI encoding matrix for dynamic MRI data
%	
%	input:
%			mask : ky-kx-t sampling mask (Ny,Nx,Nt)
%           b1 : coil sensitivity maps (Ny,Nx,Nc)
%
%	output: the operator
%
%	(c) Ricardo Otazo 2008

res.adjoint = 0;
res.mask = mask;
res.b1 = b1;
res = class(res,'Emat_xyt');

