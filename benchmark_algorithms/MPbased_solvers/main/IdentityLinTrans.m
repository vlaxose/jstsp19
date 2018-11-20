classdef IdentityLinTrans < LinTrans
    % Simple LinTrans class that implements an identity matrix.
    % Occasionally useful, avoids need to store an explicit identity matrix
    % with a MatrixLinTrans operator if one is desired.
    
    properties
        M; %identity matrix dimension
        scale; %scale factor
    end
    
    methods
        
        % Constructor
        function obj = IdentityLinTrans(M,scale)
            obj = obj@LinTrans;
            obj.M = M;
            obj.scale = scale;
        end
        
        % size method ( deals with optional dimension argin  ; nargout={0,1,2} )
        function [m,n] = size(obj,dim)
            if nargin>1 % a specific dimension was requested
                if dim>2
                    m=1;
                else
                    m=obj.M;
                end
            elseif nargout<2  % all dims in one output vector
                m=[obj.M obj.M];
            else % individual outputs for the dimensions
                m=obj.M;
                n=obj.M;
            end
        end
        
        % Matrix multiply
        function y = mult(obj,x)
            y = obj.scale*x;
        end
        % Matrix multiply transpose
        function y = multTr(obj,x)
            y = obj.scale*x;
        end
        % Matrix multiply with square
        function y = multSq(obj,x)
            y = abs(obj.scale).^2*x;
        end
        % Matrix multiply transpose
        function y = multSqTr(obj,x)
            y = abs(obj.scale)^2*x;
        end
        
        
        
    end
end