% This function is used by HUT-AMP to evaluate the cost needed to compute
% the parameters of a markov random field.
%
% Coded by: Jeremy Vila, The Ohio State Univ.
% E-mail: vilaj@ece.osu.edu
% Last change: 2/25/15
% Change summary: 
%   v 1.0 (JV)- First release
%
% Version 1.0

function [ f, g, H ] = pseudoLF(params, S_HAT, N, T)
    beta = params(1);
    alpha = params(2);

    S_HAT = (S_HAT-0.5)*2;

    horizontalZ = zeros(N,T);
    verticalZ   = zeros(N,T);

    horizontalZ(:,2:end-1) = S_HAT(:,1:end-2) + S_HAT(:,3:end);
    horizontalZ(:,1) = horizontalZ(:,2);
    horizontalZ(:,end) = horizontalZ(:,end-1);

    verticalZ(2:end-1,:) = S_HAT(1:end-2,:) + S_HAT(3:end,:);
    verticalZ(1,:) = verticalZ(2,:);
    verticalZ(end,:) = verticalZ(end-1,:);

    hvZ = horizontalZ + verticalZ;
    Z = hvZ*beta - alpha;

    f = sum(sum(- S_HAT.*(Z) + log(cosh(Z))));


    g = [sum(sum(tanh(Z).*hvZ - S_HAT.*hvZ )); sum(sum(-tanh(Z) + S_HAT))];

    H = NaN*ones(2);
    H(1,1) = sum(sum((1-(tanh(Z)).^2).*(hvZ.^2)));
    H(2,2) = sum(sum((1-(tanh(Z)).^2)));
    H(1,2) = sum(sum(-(1-(tanh(Z)).^2).*(hvZ)));
    H(2,1) = H(1,2);
    
return