% NSER 	Compute the normalized support error rate (NSER) of a recovered
% support
%
%
% SYNTAX:
% rate = nser(TrueSupp, EstSupp)
%
% INPUTS:
%   TrueSupp    Indices of the true support
%   EstSupp     Indices of the estimated support
%   
%
% OUTPUTS:
%   rate        The NSER of the estimated support
%

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: ziniel.1@osu.edu
% Last change: 02/06/12
% Change summary: 
%		- Created (02/06/12; JAZ)
% Version 0.2


function rate = nser(TrueSupp, EstSupp)

%% Compute the NSER

rate = ( numel(setdiff(TrueSupp, EstSupp)) + ...
    numel(setdiff(EstSupp, TrueSupp)) ) / numel(TrueSupp);