%***********NOTE************
% The user does not need to call this function directly.  It is called
% insead by EMGMAMP.
%
% This function resizes a vector to a N by T by L vector
%
%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail Jeremy Vila at: vilaj@ece.osu.edu
% Last change: 4/4/12
% Change summary: 
%   v 1.0 (JZ)- First release
%
% Version 2.0
%%
function OUT = resize(IN, J, K, L)
        if nargin < 4
            % Assume third dimension is singleton
            L = 1;
        end
        [J1, K1, L1] = size(IN);
        
        if J1 == 1 && K1 == 1 && L1 == L
            OUT = repmat(IN, [J, K, 1]);
        elseif J1 == J && K1 == 1 && L1 == L
            OUT = repmat(IN, [1, K, 1]);
        elseif J1 == 1 && K1 == K && L1 == L
            OUT = repmat(IN, [J, 1, 1]);
        elseif J1 == J && K1 == K && L1 == L
            OUT = IN;
        elseif J1 == 1 && K1 == 1 && L1 == 1
            OUT = repmat(IN, [J, K, L]);
        elseif J1 == J && K1 == 1 && L1 == 1
            OUT = repmat(IN, [1, K, L]);
        elseif J1 == 1 && K1 == K && L1 == 1
            OUT = repmat(IN, [J, 1, L]);
        elseif J1 == J && K1 == K && L1 == 1
            OUT = repmat(IN, [1, 1, L]);
        else
            error('Incorrect size of parameter')
        end

return