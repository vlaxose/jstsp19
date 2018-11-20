% CLASS: GAMPState
% 
% HIERARCHY (Enumeration of the various super- and subclasses)
%   Superclasses: N/A
%   Subclasses: N/A
% 
% TYPE (Abstract or Concrete)
%   Concrete
%
% DESCRIPTION (High-level overview of the class)
%   This class is used to store information about the "state" of GAMP
%   after execution, which is required by certain functions of the TurboOpt
%   class in order to perform turbo inference.
%
% PROPERTIES (State variables)
%   * All properties of this class are hidden, and should not require user
%   access or manipulation *
%
% METHODS (Subroutines/functions)
%   * All methods of this class are hidden *
% 

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: zinielj@ece.osu.edu
% Last change: 08/13/14
% Change summary: 
%       - Created (12/08/11; JAZ)
%       - Added phat and pvar (08/13/14; JAZ)
% Version 1.1
%

classdef GAMPState

    properties (GetAccess = private, SetAccess = immutable, Hidden)
        xhat;
        xvar;
        rhat;
        rvar;
        shat;
        svar;
        zhat;
        zvar;
        phat;
        pvar;
    end
    
    methods (Hidden)
        % Constructor
        function obj = GAMPState(xhat, xvar, rhat, rvar, shat, svar, ...
                zhat, zvar, phat, pvar)
            if nargin < 10
                error('Insufficient number of input arguments')
            end
            obj.xhat = xhat;
            obj.xvar = xvar;
            obj.rhat = rhat;
            obj.rvar = rvar;
            obj.shat = shat;
            obj.svar = svar;
            obj.zhat = zhat;
            obj.zvar = zvar;
            obj.phat = phat;
            obj.pvar = pvar;
        end
        
        % Accessor
        function [xhat, xvar, rhat, rvar, shat, svar, zhat, zvar, phat, ...
                pvar] = getState(obj)
            xhat = obj.xhat;
            xvar = obj.xvar;
            rhat = obj.rhat;
            rvar = obj.rvar;
            shat = obj.shat;
            svar = obj.svar;
            zhat = obj.zhat;
            zvar = obj.zvar;
            phat = obj.phat;
            pvar = obj.pvar;
        end
        
    end % methods
    
end % classdef