function A = genSparseMat(nz,nx,d)
% genSparseMat:  Generates a sparse zero-one matrix of dimensions nz x nx
% with row degree d.

nzmax = d*nx;   % number of non-zero entries
Irow = zeros(nzmax,1);
Icol = zeros(nzmax,1);
scale = sqrt(nz/d/nx);
Aval = scale*sign(rand(nzmax,1) - 0.5 );
for ix = 1:nx
    J = ((ix-1)*d+1:ix*d);
    Icol(J)=ix;
    p = zeros(d,1);
    id = 0;
    while (id < d)
        iz = ceil(nz*rand(1));
        if (all( p ~= iz))
            id = id + 1;
            p(id) = iz;
        end        
    end
    Irow(J) = p;
end
A = sparse(Irow,Icol,Aval,nz,nx,nzmax);


