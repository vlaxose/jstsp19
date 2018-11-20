classdef FxnhandleLinTrans < LinTrans
    % FxnhandleLinTrans:  Linear transform class for function handles. 
    
    properties
	M	% output dimension
	N	% input dimension
    	A	% function handle for forward multiply
    	Ah	% function handle for hermition-transpose multiply
    	S	% function handle for forward multiply-square
    	St	% function handle for transpose multiply-square
	Fro2    % 1/(M*N) times the squared Frobenius norm 
    end
    
    methods
        
        % Constructor
        function obj = FxnhandleLinTrans(M,N,A,Ah,S,St)
            obj = obj@LinTrans;

            % manditory inputs
	    if ~(isa(N,'numeric')&isa(M,'numeric'))
	          error('First and second inputs must be integers')   
	    end
            obj.M = M;
            obj.N = N;
	    if ~(isa(A,'function_handle')&isa(Ah,'function_handle'))
	          error('Third and fourth inputs must be function handles')   
	    end
            obj.Ah = Ah;
            obj.A = A;

            % optional inputs 
            if nargin > 4
                if isa(S,'double')&&(S>0)
                  % 5th input "S" contains Fro2
		  obj.Fro2 = S;
                elseif (nargin > 5)&&(isa(S,'function_handle')&isa(St,'function_handle'))
                  % 5th and 6th inputs are both function handles, S and St
                  obj.S = S;
                  obj.St = St;
	        else
	          error('Problem with the 5th & 6th inputs.  We need that either the fifth input is a positive number for Fro2, or that the fifth and sixth inputs are both function handles for S and St.')   
                end
	    else
                % approximate the squared Frobenius norm
		P = 10;      % increase for a better approximation
		obj.Fro2 = 0;
		for p=1:P,   % use "for" since A may not support matrices 
	          obj.Fro2 = obj.Fro2 + ...
		  	sum(abs(obj.A(randn(obj.N,1))).^2)/(P*M*N);
		end
	    end
        end
        
        % Size
        function [m,n] = size(obj)
	    n = obj.N;	
	    m = obj.M;
        end
        
        % Matrix multiply
        function y = mult(obj,x)
	    y = obj.A(x);
        end

        % Hermitian-transposed-Matrix multiply 
        function x = multTr(obj,y)
	    x = obj.Ah(y);
        end
        
        
        % Squared-Matrix multiply 
        function y = multSq(obj,x)
            if isempty(obj.Fro2)
	        y = obj.S(x);
            else
                y = ones(obj.M,1)*(obj.Fro2*sum(x,1));
            end
        end
        
        
        % Squared-Hermitian-Transposed Matrix multiply 
        function x = multSqTr(obj,y)
            if isempty(obj.Fro2)
	        x = obj.St(y);
            else
	        x = ones(obj.N,1)*(obj.Fro2*sum(y,1));
            end
        end
        
    end
end
