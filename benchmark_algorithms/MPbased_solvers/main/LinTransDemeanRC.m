classdef LinTransDemeanRC < LinTrans
    % Linear operator class with mean removal
    %
    % Given an m-by-n linear operator A, LinTransDemean creates a 
    % "demeaned" (m+2)-by-(n+2) operator Ad as follows:  
    %
    %  Suppose
    %
    %    oneN = ones(n,1)
    %    oneM = ones(m,1)
    %
    %  and
    %
    %    mu = oneM'*A*oneN/(M*N) is the global mean
    %    gam = A*oneN/N contains the means of each row
    %    col' = oneM'*(A-mu)/M contains the means of each column (after global mean removal)
    %    At = A - gam*oneN' - oneM*col' is the mean-removed matrix
    %
    %  Then it can be verified that
    %
    %    At*oneN=0, meaning that each row of At is zero-mean  
    %    oneM'*At=0', meaning that each column of At is zero-mean  
    %    col'*oneN=0, meaning that the c vector is zero-mean
    %
    %  Then defining
    %
    %    Ad = [At               b12*gam    b13*oneM ; 
    %          b21*oneN'        -b21*b12   0        ;
    %          b31*col'         0          -b31*b13 ]
    %
    %  it can be verified that if 
    %
    %    xd = [x; oneN'*x/b12; col'*x/b13] 
    % 
    %  then
    %
    %    Ad*xd = [A*x; 0; 0] 
    %
    %  Finally,
    properties
        % Base linear operator
        A;
       
        % Mean vectors and elementwise squares 
        gam;
	col;
        gam2;
	col2;

	% Scaling parameters
	b12;
	b21;
	b13;
	b31;

	% Explicit matrix implementation
	isExplicit;
	mtx;
	mtx2;
    end
   

    methods 
        % Constructor
        function obj = LinTransDemeanRC(A,isExplicit)
           
            % Base class
            obj = obj@LinTrans();
            
            % handle inputs
            if nargin>1
                obj.isExplicit = (isExplicit);
            else
                obj.isExplicit = true; % default
            end

            % Basic properties
            obj.A = A;
            [m,n] = A.size();

	    % Get global mean, column mean, row mean
	    A1 = A.mult(ones(n,1));
            mu = (ones(1,m)*A1)/(m*n);
            obj.col = A.multTr(ones(m,1))/m - conj(mu)*ones(n,1);
            obj.col2 = abs(obj.col).^2;
            obj.gam = A1/n;
            obj.gam2 = abs(obj.gam).^2;

	    % compute scaling constants
            Atildefro2 = ones(1,m)*obj.A.multSq(ones(n,1)) ...
                    -2*real( obj.gam'*A1 ...
                        + ones(1,m)*obj.A.mult(obj.col) ...
                        + (ones(1,m)*obj.gam)*(ones(1,n)*obj.col) ) ...
                    + n*(ones(1,m)*obj.gam2) + m*(ones(1,n)*obj.col2);
            obj.b12 = min(1, sqrt(Atildefro2/(n*(ones(1,m)*obj.gam2)))); % protect agains gam2~=0
            obj.b21 = sqrt(Atildefro2/(m*(n+obj.b12^2)));
            obj.b13 = sqrt(Atildefro2/(n*m));
            obj.b31 = sqrt(Atildefro2/(m*(ones(1,n)*obj.col2+obj.b13^2)));

	    % if implementation uses explicit matrices
            if obj.isExplicit
	        Atilde = obj.A.mult(eye(n)) - obj.gam*ones(1,n) - ones(m,1)*obj.col';
                obj.mtx = [Atilde,             obj.b12*obj.gam, obj.b13*ones(m,1);...
                           obj.b21*ones(1,n), -obj.b12*obj.b21, 0               ;...
                           obj.b31*obj.col',   0,              -obj.b13*obj.b31];
                obj.mtx2 = abs(obj.mtx).^2;
            end
        end       


        % Size
        function [md,nd] = size(obj)
            [m,n] = obj.A.size();
            md = m+2;
            nd = n+2;
        end


        % Matrix multiply:  zd = Ad*xd
        function zd = mult(obj,xd)

            if obj.isExplicit
	        % Explicit matrix multiply
	        zd = obj.mtx*xd;
	    else
                % Separate regular and additional components
                [m,n] = obj.A.size();
                x = xd(1:n,:);
                xr = xd(n+1,:);
                xc = xd(n+2,:);
            
                % Compute demean output
                zr = obj.b21*(ones(1,n)*x - obj.b12*xr);
                zc = obj.b31*(obj.col'*x - obj.b13*xc);
                z = obj.A.mult(x) - obj.gam*(zr/obj.b21) - ones(m,1)*(zc/obj.b31);
                zd = [z; zr; zc];
	    end
           
        end


        % Matrix multiply transpose:  xd = Ad'*sd
        function xd = multTr(obj,sd)

            if obj.isExplicit
                % Explicit matrix multiply
                xd = obj.mtx'*sd;
            else
                % Separate regular and additional components
                [m,n] = obj.A.size();
                s = sd(1:m,:);
                sr = sd(m+1,:);
                sc = sd(m+2,:);

                % Compute demean output
                xr = obj.b12*(obj.gam'*s - obj.b21*sr);
                xc = obj.b13*(ones(1,m)*s - obj.b31*sc);
                x = obj.A.multTr(s) - ones(n,1)*(xr/obj.b12) - obj.col*(xc/obj.b13);
                xd = [x; xr; xc];
            end
        end


        % Multiply by square:  pvard = abs(Ad).^2*xvard
        function pvard = multSq(obj, xvard)
           
            if obj.isExplicit
                % Explicit matrix multiply
                pvard = obj.mtx2*xvard;
            else
                % Separate regular and additional components
                [m,n] = obj.A.size();
                S = size(xvard,2);
                xvar = xvard(1:n,:);
                xvarr = xvard(n+1,:);
                xvarc = xvard(n+2,:);
            
                % Compute demean output
                pvarr = obj.b21^2*(ones(1,n)*xvar + obj.b12^2*xvarr);
                pvarc = obj.b31^2*(obj.col2'*xvar + obj.b13^2*xvarc);
                pvar = obj.A.multSq(xvar) ...
    	               - 2*real( (conj(obj.gam)*ones(1,S)).*obj.A.mult(xvar) ) ...
                       - 2*real( obj.A.mult((obj.col*ones(1,S)).*xvar) ) ...
                       + 2*real( obj.gam*(obj.col.'*xvar) ) ...
                       + ones(m,1)*(pvarc/obj.b31^2) ...
                       + obj.gam2*(pvarr/obj.b21^2);
                pvard = [pvar; pvarr; pvarc];
            end
        end
        
 
        % Matrix multiply with componentwise square transpose  
        %   rvard = (Ad.^2)'*svard
        function rvard = multSqTr(obj,svard)

            if obj.isExplicit
                % Explicit matrix multiply
                rvard = obj.mtx2'*svard;
            else
                % Separate regular and demean components
                [m,n] = obj.A.size();
                S = size(svard,2);
                svar = svard(1:m,:);
                svarr = svard(m+1,:);
                svarc = svard(m+2,:);
        
                % Compute demean output
                rvarr = obj.b12^2*(obj.gam2'*svar + obj.b21^2*svarr);
                rvarc = obj.b13^2*(ones(1,m)*svar + obj.b31^2*svarc);
                rvar = obj.A.multSqTr(svar) ...
                       - 2*real( (conj(obj.col)*ones(1,S)).*obj.A.multTr(svar) ) ...
                       - 2*real( obj.A.multTr((obj.gam*ones(1,S)).*svar) ) ...
                       + 2*real( obj.col*(obj.gam.'*svar) ) ...
                       + ones(n,1)*(rvarr/obj.b12^2) ...
                       + obj.col2*(rvarc/obj.b13^2);
                rvard = [rvar; rvarr; rvarc];
            end
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
	function outputEstd = expandOut(obj,outputEst,maxSumVal,isCmplx)

            [m,n] = obj.A.size();
            outputEstArray = cell(2,1);
            outputEstArray{1} = outputEst;
            outputEstArray{2} = DiracEstimOut(zeros(2,outputEst.numColumns()), maxSumVal, 1, isCmplx); 
            outputEstd = EstimOutConcat(outputEstArray,[m;2]);
        end;

        % expansion of xhat
	function xhatd = expandXhat(obj,xhat)

            [m,n] = obj.A.size();
	    xhatd = [xhat;...
                     ones(1,n)*xhat/obj.b12;...
                     obj.col'*xhat/obj.b13];
	end;

        % expansion of xvar
	function xvard = expandXvar(obj,xvar)

            [m,n] = obj.A.size();
 	    xvard = [xvar;...
                     (ones(1,n)*xvar)/obj.b12^2;...
                     (obj.col2'*xvar)/obj.b13^2];
%	    xvard = [xvar;...
%                    zeros(2,size(xvar,2))]; 
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
