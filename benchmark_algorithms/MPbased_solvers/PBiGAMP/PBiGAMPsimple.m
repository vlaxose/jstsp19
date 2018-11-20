%PBiGAMPsimple  A simple interface to PBiGAMP.m 
%
% DESCRIPTION:
% ------------
% The P-BiG-AMP estimation algorithm aims to estimate a pair of random 
% vectors (b,c) observed through an observation y from the Markov chain
%
%   b,c  
%   ->  z(m) = b.'*squeeze(A(m,:,:))*c + Ac(m,:)*b + Ab(m,:)*c for m=1:M  
%   ->  y 
%
% where the priors p(b),p(c) and likelihood function p(y|z) are all separable.
% Note that A has size M x length(b) x length(c), 
%           Ac has size M x length(b), and
%           Ab has size M x length(c).
% Setting Ac and/or Ab to empty is equivalent to setting them equal to zero.
%
% This routine, PBiGAMPsimple.m, is a simple interface to the more general
% routine PBiGAMP.m, which takes in an "A" of the type PBiGAMPProblem, 
% rather than an array (or tensor or sparse tensor) A.  The more general form
% facilitates computational shortcuts for various commonly encountered 
% structures in A.  For more details, see the paper:
%
% J.T. Parker and P. Schniter, "Parametric Bilinear Generalized Approximate 
% Message Passing," arXiv:1508.07575, Aug 2015.
%
%
% SYNTAX:
% -------
% [estOut,optFin,estHist] = PBiGAMPsimple(estInB,estInC,estOut,A,Ac,Ab,[opt])
%
% INPUTS:
% -------
% estInB:  An input estimator derived from the EstimIn class
%    based on the input distribution p_B(b_k).
% estInC:  An input estimator derived from the EstimIn class
%    based on the input distribution p_C(c_j).
% estInOut:  An output estimator derived from the EstimOut class
%    based on the output distribution p_{Y|Z}(y_i|z_i).
% A:  3-way measurement tensor (of type array or tensor or sptensor) 
% Ac: 2-way measurement tensor (of type array or tensor or sptensor or empty)
% Ab: 2-way measurement tensor (of type array or tensor or sptensor or empty)
% opt:  Option structure of type PBiGAMPOpt.

function [estFin,optFin,estHist] = PBiGAMPsimple(estInB,estInC,estOut,A,Ac,Ab,opt)

%Handle inputs
if nargin<6
    error('Need at least 6 inputs')
elseif nargin==6
    opt = PBiGAMPOpt; % load default options
elseif nargin>7
    error('Supports at most 7 inputs')
end %if nargin

%Convert "A" to tensor and check size
if isa(A,'double')
    A = tensor(A);
elseif ~(isa(A,'tensor')||isa(A,'sptensor'))
    error('A must be of type double, tensor, or sptensor')
end
if ndims(A)~=3
    error('A must be three dimensional')
end

%Convert Ac to tensor and check size
if ~isempty(Ac) 
    if isa(Ac,'double')
        Ac = tensor(Ac);
    elseif ~(isa(Ac,'tensor')||isa(Ac,'sptensor'))
        error('Ac must be of type double, tensor, or sptensor')
    end
    if ndims(Ac)~=2
        error('Ac must be two dimensional')
    end
    if (size(A,1)~=size(Ac,1))||(size(A,2)~=size(Ac,2))
        error('Sizes of Ac and A are inconsistent')
    end
end

%Convert Ab to tensor and check size
if ~isempty(Ab) 
    if isa(Ab,'double')
        Ab = tensor(Ab);
    elseif ~(isa(Ab,'tensor')||isa(Ab,'sptensor'))
        error('Ab must be of type double, tensor, or sptensor')
    end
    if ndims(Ab)~=2
        error('Ab must be two dimensional')
    end
    if (size(A,1)~=size(Ab,1))||(size(A,3)~=size(Ab,2))
        error('Sizes of Ab and A are inconsistent')
    end
end

%Build ParametricZ object
if isempty(Ab)
    if isempty(Ac)
        zObject = Affine_ParametricZ(A);
    else
        zObject = Affine_ParametricZ(A,Ac);
    end
else % ~isempty(Ab)
    if ~isempty(Ac)
        zObject = Affine_ParametricZ(A,Ac,Ab);
    else
        error('Affine_ParametricZ does not support Ac==[] and Ab~=[].  This limitation can be circumvented by swapping definitions of b and c.')    
    end
end

%Create the PBiGAMPProblem object
problem = PBiGAMPProblem();
problem.M = size(A,1);
problem.Nb = size(A,2);
problem.Nc = size(A,3);
problem.zObject = zObject;

%Call PBiGAMP.m
if (nargout>2)
    [estFin,optFin,estHist] = PBiGAMP(estInB, estInC, estOut, problem, opt);
else
    [estFin,optFin] = PBiGAMP(estInB, estInC, estOut, problem, opt);
end
