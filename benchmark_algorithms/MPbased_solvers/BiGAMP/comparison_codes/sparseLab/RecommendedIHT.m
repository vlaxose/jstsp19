function beta = RecommendedIHT(X,Y,nsweep,tol, xinitial, far,relaxation_p)

% function beta=RecommendedIHT(X,y,nsweep,tol, xinitial,far,relaxation_p)
% The goal of this function is to solve a Compressed Sensing problem with a
% very fast algorithm called RecommendedIHT. For more information you may
% refer to the paper,
% "Optimally tuned iterative reconstruction algorithms for compressed
% sensing," by Arian Maleki and David L. Donoho. 
% This function gets the measurement matrix (X) and the measurements (Y) and
% applies the IHT algorithm whose parameters are optimally tuned to the problem.
% Inputs:
%          X  : Measurement matrix; We assume that all the columns have
%               almost equal $\ell_2$ norms. The tunning has been done on
%               matrices with unit column norm. 
%           y  : output vector
%       nsweep : maximum number of iterations in the algorithm. Default
%                value is 300;
%          tol : if the relative prediction error i.e. ||Y-Ax||_2/ ||Y||_2 <
%               tol the algorithm will stop. If not provided the default
%               value is zero and tha algorithm will run for nsweep
%               iterations.
%     xinitial : This is an optional parameter. It will be used as an
%                initialization of the algorithm. All the results mentioned
%                in the paper have used initialization at the zero vector
%                which is our default value. For default value you can enter
%                just 0 as well. 
%      far     : This is a again an optional parameter. If not given the
%                algorithm will use the default optimal values. It specifies
%                the False alarm rate given to the algorithm. For the
%                default value you may also use 0;
% relaxation_p : This is also an optional parameter; If not provided the 
%                optimal vaulue of the relaxation parameter will be used instead. 
% Outputs:
%       beta   :  the estimated coeffs.
%
%  References:
% For more information about this algorithm and to see the other proposals
% of IHT please refer to the paper mentioned above and the references of 
% that paper.


% Normalization and parameters.
colnorm=mean((sum(X.^2)).^(.5));
X=X./colnorm;
Y=Y./colnorm;
[n,p]=size(X);
delta=n/p;


% default values
if nargin<3
    nsweep=300;
end
if nargin<4
    tol=0.00001;
end
if (nargin<5) | (xinitial==0)
    xinitial=zeros(p,1);
end

if nargin<6 | far==0
  if delta<.85;
    far=.0465*(delta.^2)+.0041*delta+.0012;
  else
    far=.04; 
  end
end

if nargin<7
relaxation_p=.65;   % Optimal value of relaxation.
end

x1 = xinitial;
lambda =  norminv(1-far/2,0,1);


for sweep=1:nsweep,
    % estimate noise
    r        = Y - X*x1;
    c        = abs(X'*r);
    sigmaest = relaxation_p*median(c)/.6745;
    %
    x2 = x1 + relaxation_p*X'*r;
    x1 = (abs(x2) >= lambda.*sigmaest) .* x2;
    if norm(Y-X*x1)/norm(Y)< tol
        break
    end
end

beta=x1;
