classdef LinTransDemeanRC_old < LinTrans
    % Linear operator class with mean removal
    %
    % Given an m x n linear operator A, LinTransDemean creates a 
    % "demeaned" operator (m+1) x (n+1) Ad as follows:  
    %
    %  Suppose
    %
    %    oneN = ones(n,1)/sqrt(n)
    %    oneM = ones(m,1)/sqrt(m)
    %    A = AA + mu*oneM*oneN' + row*oneN' + oneM*col'
    %
    %  where
    %
    %    AA*oneN=0, oneM'*AA=0, oneM'*row=0, col*oneN=0
    %
    %  Then defining
    %
    %    Ad = [AA               gam  oneM          ; with gam = row + mu*oneM
    %          oneN'            -1   0             ;
    %          col'/norm(col)   0    -1/norm(col)  ]
    %
    %  it can be verified that if 
    %
    %    xd = [x; oneN*x; c'*x] 
    % 
    %  then
    %
    %    Ad*xd = [z; 0; 0] 
    %
    properties
        % Base linear operator
        A;
       
        % Useful constants 
	oneN;
	oneM;

        % Mean vectors and elementwise squares 
        gam;
	col;
        gam2;
	col2;

	% Norm of col
	cc
    end
   

    methods 
        % Constructor
        function obj = LinTransDemeanRC_old(A)
            
            % Base class
            obj = obj@LinTrans();
            
            % Basic properties
            obj.A = A;
            [m,n] = A.size();
	    obj.oneN = ones(n,1)/sqrt(n);
	    obj.oneM = ones(m,1)/sqrt(m);

	    % Get global mean, column mean, row mean
	    mu = obj.oneM'*A.mult(obj.oneN);
	    obj.col = A.multTr(obj.oneM) - conj(mu)*obj.oneN;
	    obj.col2 = abs(obj.col).^2;
	    obj.cc = norm(obj.col);
	    obj.gam = A.mult(obj.oneN);
	    obj.gam2 = abs(obj.gam).^2;
        end       


        % Size
        function [md,nd] = size(obj)
            [m,n] = obj.A.size();
            md = m+2;
            nd = n+2;
        end


        % Matrix multiply:  zd = Ad*xd
        function zd = mult(obj,xd)

            % Separate regular and additional components
            [m,n] = obj.A.size();
            x = xd(1:n,:);
            xr = xd(n+1,:);
            xc = xd(n+2,:);
            
            % Compute demean output
            zr = obj.oneN'*x - xr;
            zc = (obj.col'*x - xc)/obj.cc;
            z = obj.A.mult(x) - obj.gam*zr - obj.oneM*zc*obj.cc;
            zd = [z; zr; zc];
        end


        % Matrix multiply transpose:  xd = Ad'*sd
        function xd = multTr(obj,sd)
            % Separate regular and additional components
            [m,n] = obj.A.size();
            s = sd(1:m,:);
            sr = sd(m+1,:);
            sc = sd(m+2,:);

            % Compute demean output
	    xr = obj.gam'*s - sr;
            xc = obj.oneM'*s - sc/obj.cc;
	    x = obj.A.multTr(s) - obj.oneN*xr - obj.col*xc;
            xd = [x; xr; xc];
        end


        % Multiply by square:  pvard = abs(Ad).^2*xvard
        function pvard = multSq(obj, xvard)
            
            % Separate regular and additional components
            [m,n] = obj.A.size();
            S = size(xvard,2);
            xvar = xvard(1:n,:);
            xvarr = xvard(n+1,:);
            xvarc = xvard(n+2,:);
            
            % Compute demean output
            pvarr = mean(xvar,1) + xvarr;
            pvarc = (obj.col2'*xvar + xvarc)/obj.cc^2;
	    pvar = obj.A.multSq(xvar) ...
	    	- (2/sqrt(n))*real( (conj(obj.gam)*ones(1,S)).*obj.A.mult(xvar) ) ...
		- (2/sqrt(m))*real( obj.A.mult((obj.col*ones(1,S)).*xvar) ) ...
		+ (2/sqrt(m*n))*real( obj.gam*(obj.col.'*xvar) ) ...
		+ ones(m,1)*(obj.cc^2*pvarc/m) ...
		+ obj.gam2*pvarr;
            pvard = [pvar; pvarr; pvarc];        
        end
        
 
        % Matrix multiply with componentwise square transpose  
        %   rvard = (Ad.^2)'*svard
        function rvard = multSqTr(obj,svard)

            % Separate regular and demean components
            [m,n] = obj.A.size();
            S = size(svard,2);
            svar = svard(1:m,:);
            svarr = svard(m+1,:);
            svarc = svard(m+2,:);
        
            % Compute demean output
            rvarr = obj.gam2'*svar + svarr;
            rvarc = mean(svar,1) + svarc/obj.cc^2;
	    rvar = obj.A.multSqTr(svar) ...
		- (2/sqrt(m))*real( (conj(obj.col)*ones(1,S)).*obj.A.multTr(svar) ) ...
	    	- (2/sqrt(n))*real( obj.A.multTr((obj.gam*ones(1,S)).*svar) ) ...
		+ (2/sqrt(m*n))*real( obj.col*(obj.gam.'*svar) ) ...
		+ ones(n,1)*(rvarr/n) ...
		+ obj.col2*rvarc;
	    rvard = [rvar; rvarr; rvarc];
        end


        % expansion of the input estimator
	function inputEstd = expandIn(obj,inputEst)

            [m,n] = obj.A.size();
            inputEstArray = cell(2,1);
            inputEstArray{1} = inputEst;
            inputEstArray{2} = NullEstimIn(0,0);
            inputEstd = EstimInConcat(inputEstArray,[n;2]);
        end;

        % expansion of the output estimator
	function outputEstd = expandOut(obj,outputEst)

            [m,n] = obj.A.size();
            outputEstArray = cell(2,1);
            outputEstArray{1} = outputEst;
            outputEstArray{2} = DiracEstimOut(zeros(2,outputEst.numColumns())); 
            outputEstd = EstimOutConcat(outputEstArray,[m;2]);
        end;

        % expansion of xhat
	function xhatd = expandXhat(obj,xhat)

	      xhatd = [xhat;...
                       obj.oneN'*xhat;...
                       obj.col'*xhat];
	end;

        % expansion of xvar
	function xvard = expandXvar(obj,xvar)

%	      xvard = [xvar;...
%                       mean(xvar,1);...
%                       obj.col2.'*xvar];
 	      xvard = [xvar;...
                       zeros(2,size(xvar,2))]; 
	end;

        % expansion of shat
	function shatd = expandShat(obj,shat)

	      shatd = [shat;...
                       zeros(2,size(shat,2))];% CHECK THIS!!
	end;

        % expansion of svar
	function svard = expandSvar(obj,svar)

	      svard = [svar;...
                       zeros(2,size(svar,2))];% CHECK THIS!!
	end;

        % contraction of any vector
	function vec_out = contract(obj,vec_in)

	      vec_out = vec_in(1:end-2,:);
	end;

    end

end
