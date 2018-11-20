classdef UnifVarLinTrans
    % UnifVarLinTrans:  A wrapper class that takes a generic LinTrans 
    % object and returns a new object where both the multSq and multSqTr
    % methods employ pre-processing that replaces specified elements of 
    % the input vector with identical copies of the average value, as
    % was as post-processing that replaces specified elements of the output
    % vector with identical copies of the average value.  These methods are
    % used to implement the "uniform variance" version of GAMP.  The
    % benefit is that the averaged outputs can be calculated from the 
    % averaged inputs via scalar multiplication by the squared Frobenius norm.
    properties (Access = private)
        % Original LinTrans object
        lt;
       
        % Indices
	I_avg;	% output indices to average
	I_noavg;% output indices to leave unaveraged
	m_avg;  % number of unaveraged output indices 
	m_noavg;% number of unaveraged output indices 
	J_avg;	% input indices to average
	J_noavg;% input indices to leave unaveraged
	n_avg;  % number of unaveraged input indices 
	n_noavg;% number of unaveraged input indices 

	s11; % normalizing scalar for submatrix of averaged indices
        s12; % Squared row-vector of i-averaged and j-non-averaged indices
        s21; % Squared col-vector of i-non-averaged and j-averaged indices
        S22; % Squared submatrix of non-averaged indices
    end
        
    methods 
        % Constructor
        function obj = UnifVarLinTrans(lt,I_avg,J_avg)

	    % Basic properties
            obj.lt = lt;
	    [m,n] = lt.size();

	    % Configure averaging on input side
            if (nargin>1)&&(~isempty(I_avg))
                if (min(I_avg)>=1)||(max(I_avg)<=m)
	            if length(I_avg)<m  % partial averaging
                        obj.I_avg = I_avg;
                        obj.I_noavg = setdiff(1:m,I_avg);
		    else % trivial case: no partial averaging
	                obj.I_avg = 1:m;
		        obj.I_noavg = [];
		    end;
                    obj.m_avg = length(I_avg);
                    obj.m_noavg = length(obj.I_noavg);
		else 
                    error('I_avg out of valid range')
		end 
	    else % default case: no partial averaging
	        obj.I_avg = 1:m;
		obj.m_avg = m;
		obj.I_noavg = [];
		obj.m_noavg = 0;
            end

	    % Configure averaging on input side
            if (nargin>2)&&(~isempty(J_avg))
                if (min(J_avg)>=1)||(max(J_avg)<=n)
	            if length(J_avg)<n  % partial averaging
                        obj.J_avg = J_avg;
                        obj.J_noavg = setdiff(1:n,J_avg);
		    else % trivial case: no partial averaging
	                obj.J_avg = 1:n;
		        obj.J_noavg = [];
		    end;
                    obj.n_avg = length(J_avg);
                    obj.n_noavg = length(obj.J_noavg);
		else 
                    error('J_avg out of valid range')
		end 
	    else % default case: no partial averaging
	        obj.J_avg = 1:n;
		obj.n_avg = n;
		obj.J_noavg = [];
		obj.n_noavg = 0;
            end

            % Compute s11
	    x = zeros(n,1); x(obj.J_avg) = 1;
	    tmp = lt.multSq(x);
            obj.s11 = sum(tmp(obj.I_avg))/(obj.n_avg*obj.m_avg);

            % Compute s12 and S22
	    A2 = zeros(m,obj.n_noavg);
	    for k=1:obj.n_noavg,
	      e = zeros(n,1); e(obj.J_noavg(k)) = 1;
              A2(:,k) = lt.mult(e);
	    end;
	    obj.s12 = mean(abs(A2(obj.I_avg,:)).^2,1);
	    obj.S22 = abs(A2(obj.I_noavg,:)).^2;

            % Compute s21
	    A1 = zeros(obj.m_noavg,n);
	    for k=1:obj.m_noavg,
	      e = zeros(m,1); e(obj.I_noavg(k)) = 1;
              A1(k,:) = lt.multTr(e)';
	    end;
	    obj.s21 = mean(abs(A1(:,obj.J_avg)).^2,2);

        end
        
        % Size
        function [m,n] = size(obj)            
            [m,n] = obj.lt.size();
        end
        
        % Matrix multiply:  z = A*x
        function z = mult(obj,x)
            z = obj.lt.mult(x);
        end        

        % Matrix multiply transpose:  x = A'*z
        function x = multTr(obj,z)            
            x = obj.lt.multTr(z);        
        end

        % Matrix multiply with square:  z = (abs(A).^2)*x
        function zvar = multSq(obj,xvar)
            [m,n] = obj.lt.size();

            zvar = zeros(m,size(xvar,2));
    	    zvar(obj.I_avg,:) = ones(obj.m_avg,1)*( ...
                obj.s11*sum(xvar(obj.J_avg,:),1) ...
		+ obj.s12*xvar(obj.J_noavg,:) );
	    zvar(obj.I_noavg,:) = obj.s21*sum(xvar(obj.J_avg,:),1) ...
		+ obj.S22*xvar(obj.J_noavg,:);
        end

        % Matrix multiply with componentwise square transpose:  
        % x = (abs(A).^2)'*z
        function xvar = multSqTr(obj,zvar)
            [m,n] = obj.lt.size();

            xvar = zeros(n,size(zvar,2));
	    xvar(obj.J_avg,:) = ones(obj.n_avg,1)*( ...
		obj.s11*sum(zvar(obj.I_avg,:),1) ...
		+ obj.s21'*zvar(obj.I_noavg,:) );
	    xvar(obj.J_noavg,:) = obj.s12'*sum(zvar(obj.I_avg,:),1) ...
		+ obj.S22'*zvar(obj.I_noavg,:);
        end

        % Contract for de-mean
        function x = contract(obj,x)            
            x = obj.lt.contract(x);
        end
        
    end
        
end
