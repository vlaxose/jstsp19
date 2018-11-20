%Compare two cost strategies for Matrix Completion

%Start with a clean slate
clear all %#ok<CLSCR>
clc


%Add needed paths for MC examples
setup_MC



%Handle random seed
defaultStream = RandStream.getGlobalStream;
if 1 %change to zero to try the same random draw repeatedly
    savedState = defaultStream.State;
    save random_state.mat savedState;
else
    load random_state.mat %#ok<UNRCH>
end
defaultStream.State = savedState;


%Problem parameters
SNR = 40; %Signal to Noise Ratio in dB (set to inf for noise free)
M = 500; %Data matrix is M x L
L = 500; %L >= M for some codes (doesn't matter for BiG-AMP)
N = 10; %The rank
p1 = 0.1; %fraction of observed entries

%Check condition
checkVal = (M + L).*N ./ (p1 .* M .* L);
disp(['Check condition was ' num2str(checkVal)])
rho = N*(L + M - N) / p1 / M / L;
disp(['Rho was ' num2str(rho)])


%% Define options

%Set options
opt = BiGAMPOpt; %initialize the options object

%Use sparse mode for low sampling rates
if p1 <= 0.2
    opt.sparseMode = 1;
end


%% Build the true low rank matrix

%Compute true input vector
X = randn(N,L);


%Build true A
A = randn(M,N);

%Noise free signal
Z = A*X;


%% Form the (possibly noisy) output channel


%Define the error function for computing normalized mean square error
error_function = @(qval) 20*log10(norm(qval - Z,'fro') / norm(Z,'fro'));
opt.error_function = error_function;

%Determine nuw
nuw = norm(reshape(Z,[],1))^2/M/L*10^(-SNR/10);

%Noisy output channel
Y = Z + sqrt(nuw)*randn(size(Z));

%Censor Y
omega = false(M,L);
ind = randperm(M*L);
omega(ind(1:ceil(p1*M*L))) = true;
Y(~omega) = 0;

%Specify the problem setup for BiG-AMP, including the matrix dimensions and
%sampling locations. Notice that the rank N can be learned by the EM code
%and does not need to be provided in that case. We set it here for use by
%the low-level codes which assume a known rank
problem = BiGAMPProblem();
problem.M = M;
problem.N = N;
problem.L = L;
[problem.rowLocations,problem.columnLocations] = find(omega);

%% Establish the channel objects for BiG-AMP

%Prior on X
gX = AwgnEstimIn(0, 1);

%Prior on A
gA = AwgnEstimIn(0, 1);


%Output log likelihood
if opt.sparseMode
    gOut = AwgnEstimOut(reshape(Y(omega),1,[]), nuw);
else
    gOut = AwgnEstimOut(Y, nuw);
end



%% Control initialization

%Random initializations
Ahat = randn(M,N);
xhat = randn(N,L);


%Use the initial values
opt.xhat0 = xhat;
opt.Ahat0 = Ahat;
opt.Avar0 = 1e-3*ones(M,N);
opt.xvar0 = 1e-3*ones(N,L);


%Initialize results as empty
results = [];


%% Run BiG-AMP with original cost function
opt.diagnostics = true;
disp('Starting BiG-AMP (log likelihood)')
%Run BGAMP
opt.adaptStepBethe = false;
tstart = tic;
[estFin,~,estHist] = ...
    BiGAMP(gX, gA, gOut, problem, opt);
tGAMP = toc(tstart);


%% Run BiG-AMP with Bethe cost function


disp('Starting BiG-AMP (Bethe)')
%Run BGAMP
opt.adaptStepBethe = true;
tstart = tic;
[estFin2,~,estHist2] = ...
    BiGAMP(gX, gA, gOut, problem, opt);
tGAMP2 = toc(tstart);


%% Show results

figure(1)
clf
semilogy((estHist.val),'r-o','linewidth',1);
hold on
semilogy((estHist2.val),'b-+','linewidth',1);
set(gca,'FontSize',18)
hold off
axis tight
grid on
list = {'BiGAMP loglik','BiGAMP Bethe'};
legend(list,'fontsize',15,'Location','SouthEast')
xlabel('iter')
ylabel('cost')

figure(2)
clf
plot(estHist.errZ,'r-o','linewidth',1);
hold on
plot(estHist2.errZ,'b-+','linewidth',1);
set(gca,'FontSize',18)
hold off
axis tight
grid on
list = {'BiGAMP loglik','BiGAMP Bethe'};
legend(list,'fontsize',15,'Location','NorthEast')
xlabel('iter')
ylabel('Error (NMSE)')

figure(3)
clf
plot(estHist.step,'r-o','linewidth',1);
hold on
plot(estHist2.step,'b-+','linewidth',1);
set(gca,'FontSize',18)
hold off
axis tight
grid on
list = {'BiGAMP loglik','BiGAMP Bethe'};
legend(list,'fontsize',15,'Location','SouthEast')
xlabel('iter')
ylabel('Step Size')


figure(4)
clf
alpha = estHist2.zvarMean ./ estHist2.pvarMean;
Nplot = length(alpha);
h = plotyy(1:Nplot,alpha,1:Nplot,estHist2.step);
ylabel(h(1),'approximation to average \alpha')
ylabel(h(2),'step size')
xlabel('Iteration')
grid

