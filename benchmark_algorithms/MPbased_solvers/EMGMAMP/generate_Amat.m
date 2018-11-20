%GENERATE A Matrix      Generates a real or complex matrix of size M by N 
%of a designed type.
%
%SYNTAX
% Amat = generate_Amat(Params)
%
% INPUTS:
% Params            A structure containing signal model parameters
%   .N              The dimension of the true signal, x
%   .M              The dimension of the measurement vector, y
%   .type           Type of matrix to generate (1) for an i.i.d (complex) 
%                   Gaussian matrix.  (2) For an i.i.d real Rademacher
%                   matrix (3) for a oversampled dft. [default 1]
%   .realmat        Specify whether matrix is real valued [default true]
%   .tau            Specify correlation among columns of A between 0 and 1
%                   Default is 0.  If turned on it is recommended to set
%                   GAMP options GAMPopt.adaptStep = true, GAMPopt.bbStep =
%                   true. This options only applies to Gaussian matrices.
%   .normcolumns    Decide whether to normalize the columns of the matrix.
%                   [default = true]
%
%OUTPUTS
%   Amat            The resulting A matrix
%
% Coded by: Jeremy Vila, The Ohio State Univ.
% E-mail: vilaj@ece.osu.edu
% Last change: 4/4/12
% Change summary: 
%   v 1.0 (JV)- First release
%   v 1.3 (JV)- Allows for creation of DFT operator
%   v 2.0 (JV)- Accounts for MMV model in DFT multiplication.
%
% Version 2.0
function Amat = generate_Amat(Params)

%%
M = Params.M; N = Params.N;

if ~isfield(Params,'type')
    Params.type = 1;
end

if ~isfield(Params,'normcolumns')
    Params.normcolumns = true;
end

if ~isfield(Params,'realmat')
    Params.realmat = true;
end

if Params.type == 1;
    
    if ~isfield(Params,'tau')
        Params.tau = 0;
    end
    
    %Compute initial variance
    var0 = (1 - Params.tau)^2 / (1 - Params.tau^2); %initial variance

    %Do the AR model manually- actually faster when M is large
    Amat = zeros(M,N);
    if Params.realmat == true
        Amat(:,1) = sqrt(var0)*(randn(M,1));
        for kk = 2:N
            Amat(:,kk) = Params.tau*Amat(:,kk-1) ...
                +  (1 - Params.tau)*sqrt(1)*(randn(M,1));
        end
    else
        Amat(:,1) = sqrt(var0/2)*(randn(M,1)+1i*randn(M,1));
        for kk = 2:N
            Amat(:,kk) = Params.tau*Amat(:,kk-1) ...
                +  (1 - Params.tau)*sqrt(1/2)*(randn(M,1)+1i*randn(M,1));
        end 
    end

elseif Params.type == 2
    Amat = sign(randn(M,N));
else
    %Build the object
    Amat = FourierLinTrans(N,N,1);

    %Compute random sample locations
    Amat.ySamplesRandom(M);
end

%Normalize the columns
if Params.normcolumns == true && Params.type ~= 3;
    columnNorms = sqrt(diag(Amat'*Amat));
    Amat = Amat*diag(1 ./ columnNorms); %unit norm columns
end

return