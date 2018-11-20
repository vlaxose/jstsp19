function  res = Emat_xy(mask,b1)

res.adjoint = 0;
res.mask = mask;
res.b1=b1;
res = class(res,'Emat_xy');

