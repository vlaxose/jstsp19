%% Evaluating the code in this cell will plot the *input* channel 
%% thresholding function as the argument 'rhat' is swept over a range of 
%% values, with all other arguments and parameters held fixed

% choose parameters
prior = 'laplacian'; % e.g., 'bernoulli-gaussian' or 'laplacian' or 'bernoulli-ellp' or
                        % 'elastic'
maxSum = 1;	% MAP (or max-sum) versus MMSE (or sum-product)
p1 = 0.2;	% "on" probability for bernoulli-* priors
rmax = 2;	% range of input mean
rvar = 0.25;	% input variance 

% build title
tit_str = prior;
if maxSum,
  tit_str = ['MAP ',tit_str];
else
  tit_str = ['MMSE ',tit_str];
end;
tit_str = [tit_str,', rvar=',num2str(rvar)];
if strfind(prior,'bernoulli'),
  tit_str = [tit_str,', p1=',num2str(p1)];
end;

% configure threshold
if strfind(prior,'gaussian')  
  xhat0 = 0;
  xvar0 = 1;
  inputEst0 = AwgnEstimIn(xhat0,xvar0,maxSum);
  tit_str = [tit_str,', xhat0=',num2str(xhat0),', xvar0=',num2str(xvar0)];

elseif strfind(prior,'laplacian')  
  lambda = 1;
  inputEst0 = SoftThreshEstimIn(lambda, maxSum);
  tit_str = [tit_str,', lambda=',num2str(lambda)];

elseif strfind(prior,'ellp')  
  if maxSum~=1, error('MMSE ell-p not yet implemented!'); end;
  lambda = 0.5;
  p = 1.0;
  inputEst0 = EllpEstimIn(lambda,p);
  tit_str = [tit_str,', p=',num2str(p),', lambda=',num2str(lambda)];
  
elseif strfind(prior, 'elastic')
    lambda1 = 1e0;
    lambda2 = 1e0;
    inputEst0 = ElasticNetEstimIn(lambda1, lambda2, logical(maxSum));
    tit_str = [tit_str, ', lambda1=',num2str(lambda1) ', lambda2=',num2str(lambda2)];

else
  error('Prior not recognized!')
end

% sparsify
if strfind(prior,'bernoulli'),
  inputEst = SparseScaEstim(inputEst0,p1,0,maxSum);
else 
  inputEst = inputEst0;
end

% compute input-output relationship
rhat = linspace(-rmax,rmax,1e5);
[xhat,xvar,val] = inputEst.estim(rhat,rvar*ones(size(rhat)));


% plot input-output relationship
clf;
subplot(121)
 handy = plot(rhat,xhat);
 set(handy,'Linewidth',2)
 hold on; plot(rmax*[-1,1],rmax*[-1,1],'r--'); hold off;
 xlabel('rhat');
 ylabel('xhat');
 axis('equal')
 grid on;
subplot(122)
 handy = plot(rhat,xvar);
 set(handy,'Linewidth',2)
 hold on; plot(rmax*[-1,1],rvar*[1,1],'r--'); hold off;
 xlabel('rhat');
 ylabel('xvar');
 grid on;
axes('Position',[0 0 1 1],'Visible','off');
 handy = text(0.5,0.96,tit_str);
 set(handy,'horizontalalignment','center')

 
 %% Evaluating the code in this cell will plot the *output* channel 
%% thresholding function as the argument 'phat' is swept over a range of 
%% values, with all other arguments and parameters held fixed

% choose parameters
likelihood = 'logit'; % e.g., 'awgn', 'probit', 'logit' data likelihood, p(y|z)
maxSum = 0;     % MAP (or max-sum) versus MMSE (or sum-product)
pmax = 2;       % range of input mean argument (phat \in [-pmax, pmax])
pvar = 0.25;	% input variance
Nval = 1e3;     % # of samples of threshold to evaluate

% build title
tit_str = likelihood;
if maxSum,
  tit_str = ['MAP ',tit_str];
else
  tit_str = ['MMSE ',tit_str];
end;
tit_str = [tit_str,', pvar=',num2str(pvar)];
% Create two separate title string copies
tit_str1 = tit_str;
tit_str2 = tit_str;

% configure threshold
if strfind(likelihood,'awgn')  
    wvar = 1e-1;    % Noise variance
    
    y = -1*ones(Nval,1);	% Observed data value
    outputEst1 = AwgnEstimOut(y, wvar, maxSum);
    tit_str1 = [tit_str1,', y = ',num2str(y(1)),', wvar = ',num2str(wvar)];
    
    y = 1*ones(Nval,1);     % Observed data value
    outputEst2 = AwgnEstimOut(y, wvar, maxSum);
    tit_str2 = [tit_str2,', y = ',num2str(y(1)),', wvar = ',num2str(wvar)];
elseif strfind(likelihood,'probit')
    probit_var = 1e-2;      % Probit channel variance
    
    y = zeros(Nval,1);      % Observed data value
    outputEst1 = ProbitEstimOut(y, 0, probit_var, logical(maxSum));
    tit_str1 = [tit_str1,', y = ',num2str(y(1)),', probit\_var = ',num2str(probit_var)];
    
    y = ones(Nval,1);      % Observed data value
    outputEst2 = ProbitEstimOut(y, 0, probit_var, logical(maxSum));
    tit_str2 = [tit_str2,', y = ',num2str(y(1)),', probit\_var = ',num2str(probit_var)];
elseif strfind(likelihood,'logit')
    logit_scale = 1e1;      % Logit channel scale parameter
    
    y = zeros(Nval,1);      % Observed data value
    outputEst1 = LogitEstimOut(y, logit_scale, [], [], logical(maxSum));
    tit_str1 = [tit_str1,', y = ',num2str(y(1)),', logit\_scale = ',num2str(logit_scale)];
    
    y = ones(Nval,1);      % Observed data value
    outputEst2 = LogitEstimOut(y, logit_scale, [], [], logical(maxSum));
    tit_str2 = [tit_str2,', y = ',num2str(y(1)),', logit\_scale = ',num2str(logit_scale)];
else
  error('Likelihood model not recognized!')
end

% compute input-output relationship
phat = linspace(-pmax,pmax,Nval)';
[zhat1, zvar1] = outputEst1.estim(phat,pvar*ones(size(phat)));
[zhat2, zvar2] = outputEst2.estim(phat,pvar*ones(size(phat)));

% plot input-output relationship
clf;
subplot(221)
handy = plot(phat,zhat1);
set(handy,'Linewidth',2)
hold on; plot(pmax*[-1,1],pmax*[-1,1],'r--'); hold off;
xlabel('phat');
ylabel('zhat');
axis('tight')
grid on;
subplot(222)
handy = plot(phat,zvar1);
set(handy,'Linewidth',2)
hold on; plot(pmax*[-1,1],pvar*[1,1],'r--'); hold off;
xlabel('phat');
ylabel('zvar');
grid on;
axes('Position',[0 0 1 1],'Visible','off');
subplot(223)
handy = plot(phat,zhat2);
set(handy,'Linewidth',2)
hold on; plot(pmax*[-1,1],pmax*[-1,1],'r--'); hold off;
xlabel('phat');
ylabel('zhat');
axis('tight')
grid on;
subplot(224)
handy = plot(phat,zvar2);
set(handy,'Linewidth',2)
hold on; plot(pmax*[-1,1],pvar*[1,1],'r--'); hold off;
xlabel('phat');
ylabel('zvar');
grid on;
axes('Position',[0 0 1 1],'Visible','off');
handy = text(0.5,0.96,tit_str1);
set(handy,'horizontalalignment','center')
handy = text(0.5,0.49,tit_str2);
set(handy,'horizontalalignment','center')
