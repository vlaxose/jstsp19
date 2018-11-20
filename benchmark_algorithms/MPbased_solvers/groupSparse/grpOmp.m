function xhat = grpOmp(y,A,grpInd,nit)

% Get dimensions
Aorig = A;
[nz,nx] = size(A);

% Find indices within each group
ngrp = max(grpInd);
Igrp = cell(ngrp,1);
for igrp = 1:ngrp
    Igrp{igrp} = find(grpInd == igrp);
end

% Initialize vectors
Isel = zeros(ngrp,1);
r = y;

% Main OMP loop
for it = 1:nit
    
    % Compute correlations within each group
    rho = zeros(ngrp,1);
    for igrp = 1:ngrp 
        if (Isel(igrp))     % Skip if group is already selected
            continue;
        end
        Ai = A(:,Igrp{igrp});  
        v = Ai'*r;
        rho(igrp) = v'* ((Ai'*Ai) \ v);
    end
    
    % Find maximum energy
    [mm,im] = max(rho);
    Isel(im) = 1;
    
    % Add each of the columns in the group and apply Gram-Schmidt 
    Ai = A(:,Igrp{im});
    ni = size(Ai,2);
    for icol = 1:ni
        % Get the column
        ai = Ai(:,icol);      % Get the column
        ai = ai / norm(ai);
        
        % Remove the energy in the other columns of A
        A = A - ai*(ai'*A);
        
        % Remove the energy from the residual
        r = r - ai*(ai'*r);
    end            
end

% Compute estimate
I = find(Isel(grpInd));
xhat = zeros(nx,1);
xhat(I) = Aorig(:,I) \ y;
