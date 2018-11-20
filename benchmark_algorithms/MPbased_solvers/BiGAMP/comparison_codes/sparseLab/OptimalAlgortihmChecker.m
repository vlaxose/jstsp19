function OptimalAlgortihmChecker
% function OptimalAlgortihmChecker
% In this function we show an example of how one can generate a problem
% instance and then run one of the algorithms on that problem instance to
% get the sparsest solution. 

disp(' ********* NEW *************')
p=1000*2; n=250*2; k=40*2;  % dimensions of the problem.
nsweep=500;           % total number of sweeps in our thresholding algorithm.
tol=.001;

%----------------------------
% Generating Problem Instance
%----------------------------
X  = randn(n,p); %MatrixEnsemble(n,p,'USE');        % Measurement matrix
beta = SparseVector(p, k, 'signs', 0); % Generating an sparse input vector.
Y=X*beta;                              % vector of the measurements

%------------------------------------
% Applying the recommended Algorithm;  
%------------------------------------

%*****************    IHT   **********************
fprintf('N= %i, n=%i, k=%i \n',p,n,k)
tic
beta_hat=RecommendedIHT(X,Y,nsweep,tol);
elapsed_time=toc;
fprintf('IHT:  Time Needed = %i, Prediction Error= % i \n',elapsed_time, norm(Y-X*beta_hat)/norm(Y))


%*****************   IST  *************************
tic
beta_hat=RecommendedIST(X,Y,nsweep,tol);
elapsed_time=toc;
fprintf('IST:  Time Needed = %i, Prediction Error= % i \n',elapsed_time, norm(Y-X*beta_hat)/norm(Y))


%****************    TST   ***********************
tic
beta_hat=RecommendedTST(X,Y, nsweep,tol);
elapsed_time= toc;
fprintf('TST:  Time Needed = %i, Prediction Error= % i \n',elapsed_time, norm(Y-X*beta_hat)/norm(Y))


%****************    OMP   ***********************
tic
[beta_hat,iters,activationHist]=SolveOMP(X,Y,p,5000, 0,0,0,tol);
elapsed_time= toc;
fprintf('OMP:  Time Needed = %i, Prediction Error= % i \n',elapsed_time, norm(Y-X*beta_hat)/norm(Y))


%*****************   LARS-Lasso ******************   
tic
colnorm=mean((sum(X.^2)).^(.5));
X=X./colnorm;
Y=Y./colnorm;
[beta_hat,iters,activationHist,duals]= SolveLasso(X,Y,p,'Lasso',5000, 0,0,0,0,tol);
elapsed_time = toc;
fprintf('LARS-Lasso:  Time Needed = %i, Prediction Error= % i \n',elapsed_time, norm(Y-X*beta_hat)/norm(Y))



