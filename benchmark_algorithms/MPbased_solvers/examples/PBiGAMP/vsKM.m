%Handle random seed
 if verLessThan('matlab','7.14')
   defaultStream = RandStream.getDefaultStream;
 else
   defaultStream = RandStream.getGlobalStream;
 end;
 if 1 % new random trial
   savedState = defaultStream.State;
   save random_state.mat savedState;
 else % repeat last trial
   load random_state.mat
 end
 defaultStream.State = savedState;

 
 % collect .mat files and plot results, or generate new .mat files?
 collect = 1;            

 % simulation parameters
 working_dir = '~schniter/research/sparse/PBiGAMP/matlab/';
 base_dir = 'mat/';     % base directory to save data
 dont_save = false;     % used to prevent all saving of data
success_dB = -95;	% definition of "success" in terms of NMSE-dB
 ptc = 0.5;		% success rate that defines the PTC 
 outage = 0.5;		% outage rate in [0,1] (set =0 to plot average)
 normalized_axes = true;% normalize axes or use absolute lengths?
 SNRdB = 100;  	        % SNR in dB. 

 % default algorithmic parameters
 clear optPB;
 optPB.verbose = false;
 optPB.plotFig = 0;
 optPB.meanB = 0; optPB.meanC = 0;
 optPB.varB = 1; optPB.varC = 1;
 optPB.maxTry = 20; % maximum number of tries
optPB.maxIt = 500; % max number of iterations per try [250, 500]
optPB.stepInit = 0.4; % initial value of damping parameter [0.3, 0.05]
 optPB.stepIncr = 1.006; % per-iter increase of damping parameter [1.006, 1.05]
 optPB.stepMax = 0.5; % max damping parameter [0.5]
 optPB.varGainInit = 1; % gain on variance initialization [1, 10]
 optPB.meanGainInitB = 1; % gain on mean initialization of B [1]
 optPB.meanGainInitC = 1; % gain on mean initialization of C [1]
 optPB.meanInitTypeB = 'randn'; % type of mean initialization for B ['randn']
 optPB.meanInitTypeC = 'randn'; % type of mean initialization for C ['randn']
 optPB.normTol = 1e-10; % stop when norm(z)<normTol
 optPB.tol = 1e-7; % stopping tolerance
 EMinitSNRdB = min(30,SNRdB); % inital SNRdB for EM [30]

 % load scenario
 DEMO = 5; 
 switch DEMO
   case 1 % used to test a single point
     dont_save = true; % this mode is for testing only 
     dir_str = [base_dir,'vs/']; % don't save
    trials = 1;
     alg_ = ['PBiGAMP'];
     optPB.maxTry = 20;
     affine = false;
     isreal = true; 
    EM = false;
    N = 50; 
    spar = 0.6;
    alpha = 0.9;
     K = spar*N;
     M = alpha*(2*N);
     spaceKM = 'lin';	% 'lin' or 'log2'
     Kmin = K; Kmax = K; numK = 1;
     Mmin = M; Mmax = M; numM = 1;
    optPB.verbose = true;
    optPB.plotFig = 10; % plot history

   case 2 % versus # measurements at sparsity=0.2
     dir_str = [base_dir,'vsM/'];
    trials = 100;
     alg_ = ['PBiGAMP'];
     affine = false;
     isreal = true; 
    EM = false;
    N = 100; 
    spar = 0.1;
     spaceKM = 'lin';	% 'lin' or 'log2'
     Kmin = spar*N; Kmax = spar*N; numK = 1;
     Mmin = 0.1*(2*N); Mmax = 1.1*(2*N); numM = 11;

   case 3 % Phase plane 
     dir_str = [base_dir,'vsKM/'];
    trials = 1;
     alg_ = ['PBiGAMP'];
     affine = false;
     isreal = true; 
    EM = false;
    N = 100; 
     spaceKM = 'lin';	% 'lin' or 'log2'
     Kmin = 0.1*N; Kmax = N; numK = 10;
     Mmin = 0.2*(2*N); Mmax = 1.2*(2*N); numM = 11;

   case 4 % test transition around M/(2N)=0.5 for 0.1 sparsity
     dir_str = [base_dir,'vsM/'];
    trials = 100;
     alg_ = ['PBiGAMP'];
     affine = false;
     isreal = true; 
    EM = false;
    N = 50; 
    spar = 0.1;
     spaceKM = 'lin';	% 'lin' or 'log2'
     Kmin = spar*N; Kmax = spar*N; numK = 1;
     Mmin = 0.2*(2*N); Mmax = 0.6*(2*N); numM = 17;

   case 5 % test transition around M/(2N)=0.8 for 0.6 sparsity
     %dir_str = [base_dir,'vsMfixedSNR/'];
     dir_str = [base_dir,'vsM2/'];
    trials = 10;
     alg_ = ['PBiGAMP'];
     affine = false;
     isreal = true; 
    EM = false;
    N = 200; 
    spar = 0.6;
     spaceKM = 'lin';	% 'lin' or 'log2'
     Kmin = spar*N; Kmax = spar*N; numK = 1;
     Mmin = 0.5*(2*N); Mmax = 0.9*(2*N); numM = 17;

 end


%-------------------------------------------------------------------------------

 % construct K and M vectors 
 if strcmp(spaceKM,'log2'),
   K_ = unique(round(2.^linspace(log2(Kmin),log2(Kmax),numK)));
   M_ = unique(round(2.^linspace(log2(Mmin),log2(Mmax),numM)));
 elseif strcmp(spaceKM,'lin'),
   K_ = unique(round(linspace(Kmin,Kmax,numK)));
   M_ = unique(round(linspace(Mmin,Mmax,numM)));
 else
   error('spaceKM is invalid')
 end;

 % file names
 cd(working_dir);
 alg_str = []; for a=1:size(alg_,1), alg_str = [alg_str,deblank(alg_(a,:)),'_']; end;
 if length(K_)>1, 
   if strcmp(spaceKM,'log2'),
     K_str = ['_K',num2str(K_(1)),':log2:',num2str(K_(end))];
   elseif strcmp(spaceKM,'lin'),
     K_str = ['_K',num2str(K_(1)),...
              ':',num2str(K_(2)-K_(1)),':',num2str(K_(end))];
   end;
 else
   K_str = ['_K',num2str(K_)];
 end;
 if length(M_)>1,
   if strcmp(spaceKM,'log2'),
     M_str = ['_M',num2str(M_(1)),':log2:',num2str(M_(end))];
   elseif strcmp(spaceKM,'lin'),
     M_str = ['_M',num2str(M_(1)),...
              ':',num2str(M_(2)-M_(1)),':',num2str(M_(end))];
   end;
 else
   M_str = ['_M',num2str(M_)];
 end;
 name_str = [alg_str,...
            'N',num2str(N),...
            K_str,...
            M_str,...  
            '_snr',num2str(SNRdB),...
            '_a',num2str(double(affine)),...  
	    '_r',num2str(double(isreal)),...
	    '_em',num2str(double(EM)),...
	    ];
 seed_str = ['_seed',num2str(ceil(rand(1)*1e6))];
 file_str = [dir_str,name_str,'_tr',num2str(trials),seed_str,'.mat'];
 if ~exist(dir_str,'dir') 
   error('directory does not exist');
 end
 if (~collect)&&(~dont_save),
   while exist(file_str,'file'),
     randn(1);                  % increment randn generator
     seed_str = ['_seed',num2str(ceil(rand(1)*1e6))];
     file_str = [dir_str,name_str,'_tr',num2str(trials),seed_str,'.mat'];
   end;
 end;

%-------------------------------------------------------------------------------

 % load data (collect=true) or run experiments (collect=false)
 if ~collect,

  % loop over sparsity
  %M_ = sort(M_,'descend');		% order from big to small!
  nrse = NaN*ones(length(K_),length(M_),trials,size(alg_,1));
  nmse = NaN*ones(length(K_),length(M_),trials,size(alg_,1));
  time = NaN*ones(length(K_),length(M_),trials,size(alg_,1));
  trys = NaN*ones(length(K_),length(M_),trials,size(alg_,1));
  fprintf(1,'percent complete:  0')
  for k=1:length(K_),
   K = K_(k);
 
   % loop over measurements
   for m=1:length(M_), 
    M = M_(m);
 
    % loop over realizations
    for t=1:trials,

     %% break out if this (k,m) pair is impossible
     if (M<2*K)
       break;
     end

     % Create an iid Bernoulli-Gaussian vector
     Nb = N; Nc = N; Kb = K; Kc = K;
     b = zeros(Nb,1); c = zeros(Nc,1);
     if isreal,
       b(randperm(Nb,Kb)) = randn(Kb,1);
       c(randperm(Nc,Kc)) = randn(Kc,1);
     else
       b(randperm(Nb,Kb)) = sqrt(1/2).*(randn(Kb,2)*[1;1i]);  
       c(randperm(Nc,Kc)) = sqrt(1/2).*(randn(Kc,2)*[1;1i]);  
     end;

     % Create an iid Gaussian measurement tensor
     if isreal
       A = randn(M,Nb,Nc);
       if affine
         Ac = randn(M,Nb);
         Ab = randn(M,Nc);
       else
         Ac=[]; Ab=[];
       end
     else
       A = complex(randn(M,Nb,Nc),randn(M,Nb,Nc));
       if affine
         Ac = sqrt(1/2)*complex(randn(M,Nb),randn(M,Nb));
         Ab = sqrt(1/2)*complex(randn(M,Nc),randn(M,Nc));
       else
         Ac=[]; Ab=[];
       end
     end;
     Afro2 = norm(A(:))^2;
     Abfro2 = norm(Ab(:))^2;
     Acfro2 = norm(Ac(:))^2;

     % Sparsify if desired (unfortunately this makes things super slow!)
     sparsRatA = 1; % set =1 to disable sparsification
     if (sparsRatA<1)&&(sparsRatA>0)
       A = (1/sqrt(sparsRatA))*A.*(rand(size(A))<sparsRatA);
       Afro2 = norm(A(:))^2;
       A = sptensor(A);
       if affine
         Ac = (1/sqrt(sparsRatA))*Ac.*(rand(size(Ac))<sparsRatA);
         Acfro2 = norm(Ac(:))^2;
         Ac = sptensor(Ac);
         Ab = (1/sqrt(sparsRatA))*Ab.*(rand(size(Ab))<sparsRatA);
         Abfro2 = norm(Ab(:))^2;
         Ab = sptensor(Ab);
       end
     elseif sparsRatA~=1
       error('illegal sparsRatA')
     end


     % Create a noisy measurement 
     z = double(ttv(ttv(tensor(A),c,3),b,2));
     if ~isempty(Ac), z = z + Ac*b; end;
     if ~isempty(Ab), z = z + Ab*c; end;
     wvar = norm(z)^2/M*10^(-SNRdB/10); % dynamic noise variance
     %wvar = Kb*Kc * 10^(-SNRdB/10); % fixed noise variance

     if isreal
       w = sqrt(wvar)*randn(M,1);
     else
       w = sqrt(wvar/2)*(randn(M,2)*[1;1i]);
     end
     y = z + w;

     % create error functions
     errfxnB = @(bhat) (norm(bhat*((bhat'*b)/norm(bhat)^2)-b)/norm(b))^2;
     errfxnC = @(chat) (norm(chat*((chat'*c)/norm(chat)^2)-c)/norm(c))^2;

     % loop over algs
     savedState = defaultStream.State;
     for a=1:size(alg_,1),

      % start algs from same random seed 
      defaultStream.State = savedState;

      % run alg
      if strfind(alg_(a,:),'PBiGAMP'),

       % Setup PBiGAMP
       optPB.EM = EM;
       if EM
         optPB.wvar = mean(abs(y).^2)*10^(-EMinitSNRdB/10); % guess
       else
         optPB.wvar = wvar; % true noise variance 
       end
       optPB.sparB = Kb/Nb; optPB.sparC = Kc/Nc;
       optPB.cmplx = ~isreal; 
       optPB.errTry = 10^(success_dB/10);
       optPB.errfxnB = errfxnB; % optional; for plotting
       optPB.errfxnC = errfxnC; % optional; for plotting

       % Run PBiGAMP
       tstart = tic;
         outPB = PBiGAMPiidBGawgn(y,A,Ac,Ab,optPB);
       runtime = toc(tstart);
       numTry = outPB.numTry;
       err = outPB.err;
       errB = outPB.errB;
       errC = outPB.errC;

      else
       error([alg_(a,:),' not supported']);

      end;%if strfind(alg...

      % record data
      nrse(k,m,t,a) = err;
      nmse(k,m,t,a) = 0.5*(errB+errC); % average of two NMSEs
      time(k,m,t,a) = runtime;
      trys(k,m,t,a) = numTry;

      % print progress
      fprintf(1,'\b\b%2d',floor(...
      	((((k-1)*length(M_)+(m-1))*trials+(t-1))*size(alg_,1)+a-1) ...
 	/length(K_)/length(M_)/trials/size(alg_,1)*100));

     end;% for a
    end;% for t
   end;% for m
  end;% for k
  fprintf(1,'\b\b100\n')
  if ~dont_save
    bad = cell(2,1); bad{1}='errfxnB'; bad{2}='errfxnC';
    optPB=rmfield(optPB,bad); % don't save function handles
    save(file_str,'N','K_','M_',...
        'affine','trials','SNRdB','alg_',...
	'optPB','nrse','nmse','time','trys','savedState');
    display(['saved file ',file_str])
  end
 else%if collect

  % load data from files
  wildcard = [dir_str,name_str,'*.mat'];
  files = dir(wildcard);
  if (size(files,1)>0),
    nrse_ = [];
    nmse_ = [];
    time_ = [];
    trys_ = [];
    % copy into collection variables
    for f=1:size(files,1),
      load([dir_str,files(f).name]);	% provides nmse
      nrse_ = cat(3,nrse_,nrse);
      nmse_ = cat(3,nmse_,nmse);
      time_ = cat(3,time_,time);
      if ~exist('trys'), 
        trys=nan(size(time)); warning('no variable named trys'); 
      end;
      trys_ = cat(3,trys_,trys);
    end;
    trials = size(nrse_,3);
    % rename collection variables
    nrse = nrse_; clear nrse_;
    nmse = nmse_; clear nmse_;
    time = time_; clear time_;
    trys = trys_; clear trys_;
  else
    error(['No files to collect!  Looking for ',wildcard]);
  end;

 end;%if collect

%-------------------------------------------------------------------------------

 % optionally remove certain traces
 if DEMO==4
   rmalg_=['prGAMP200';'prGAMP2  ';'prGAMP1  '];
 else
   rmalg_=[];
 end
 rmindx = [];
 for a = 1:size(rmalg_,1)
   for aa = 1:size(alg_,1)
     if strcmp(rmalg_(a,:),alg_(aa,:))
       rmindx = [rmindx,aa];
     end
   end
 end
 keepindx = setdiff([1:size(alg_,1)],rmindx);
 alg_ = alg_(keepindx,:);
 nrse = nrse(:,:,:,keepindx);
 nmse = nmse(:,:,:,keepindx);
 trys = trys(:,:,:,keepindx);
 time = time(:,:,:,keepindx);

 % option for imposing a time-limit 
 time_wall = inf;	% set to inf to disable!
 trys_wall = inf;
 if time_wall<inf, 
   display(['retroactively imposing a runtime limit of ',num2str(time_wall),' seconds!']); 
 end;
 if trys_wall<inf, 
   display(['retroactively imposing a limit of ',num2str(trys_wall),' tries!']); 
 end;
 wall_bad = (time>time_wall)|(trys>trys_wall);
 nrse(wall_bad) = inf*nrse(wall_bad);
 nmse(wall_bad) = inf*nmse(wall_bad);
 trys(wall_bad) = inf*trys(wall_bad);
 time(wall_bad) = time_wall;

 % compute average quantities
 success_thresh = 10^(success_dB/10);
 success = (nmse<=success_thresh).*(~isnan(time));
 failure = (nmse>success_thresh).*(~isnan(time));
 success_avg = squeeze(mean(success,3));
 failure_avg = squeeze(mean(failure,3));
 nrse_avg = squeeze(mean(nrse,3));
 nmse_avg = squeeze(mean(nmse,3));
 time_good_avg = NaN*ones(length(K_),length(M_),size(alg_,1));
 time_bad_avg = NaN*ones(length(K_),length(M_),size(alg_,1));
 time_ran_out = NaN*ones(length(K_),length(M_),size(alg_,1));
 for k=1:length(K_),
  for m=1:length(M_),
   for a=1:size(alg_,1),
    indx_good = find(success(k,m,:,a));
    time_good_avg(k,m,a) = mean(time(k,m,indx_good,a));
    indx_bad = find(failure(k,m,:,a));
    time_bad_avg(k,m,a) = mean(time(k,m,indx_bad,a));
    indx_ran = find(success(k,m,:,a)+failure(k,m,:,a));
    if (length(indx_ran)>0)
      if outage>0,
        tmp = sort(time(k,m,indx_ran,a));
        time_ran_out(k,m,a) = tmp(ceil(outage*length(indx_ran)));
      else
        time_ran_out(k,m,a) = mean(time(k,m,indx_ran,a));
      end;
    else
      time_ran_out(k,m,a) = nan;
    end;
   end;
  end;
 end;
 time_good_avg = squeeze(time_good_avg);
 time_bad_avg = squeeze(time_bad_avg);
 time_ran_out = squeeze(time_ran_out);

 % extract outage NMSE
 nrse_out_dB = NaN*ones(length(K_),length(M_),size(alg_,1));
 for a=1:size(alg_,1),
   if outage>0,
     tmp = sort(nrse(:,:,:,a),3);	% could add eps to avoid log(0)=-inf problem 
     nrse_out_dB(:,:,a) = 10*log10(tmp(:,:,ceil(outage*trials)));
   else
     nrse_out_dB(:,:,a) = 10*log10(mean(nrse(:,:,:,a),3));
   end;
 end;
 nmse_out_dB = NaN*ones(length(K_),length(M_),size(alg_,1));
 for a=1:size(alg_,1),
   if outage>0,
     tmp = sort(nmse(:,:,:,a),3);	% could add eps to avoid log(0)=-inf problem 
     nmse_out_dB(:,:,a) = 10*log10(tmp(:,:,ceil(outage*trials)));
   else
     nmse_out_dB(:,:,a) = 10*log10(mean(nmse(:,:,:,a),3));
   end;
 end;

 % extract outage trys
 trys_out = NaN*ones(length(K_),length(M_),size(alg_,1));
 for a=1:size(alg_,1),
   if outage>0,
     tmp = sort(trys(:,:,:,a),3);	
     trys_out(:,:,a) = tmp(:,:,ceil(outage*trials));
   else
     trys_out(:,:,a) = mean(trys(:,:,:,a),3);
   end;
 end;

%-------------------------------------------------------------------------------

 % prepare for plotting
 plot_type = 'ptc';	% either 'contour' or 'imagesc' or 'ptc'
 tit_str1 = ['N=',num2str(N),...
	    ', snr=',num2str(SNRdB),'dB',...
	    ', isreal=',num2str(isreal),...
	    ', EM=',num2str(double(EM)),...
	    ', avg=',num2str(trials)];
 if normalized_axes % plot K/N and M/2/N on axes
   strK_ = []; for k=1:length(K_), strK_ = strvcat(strK_,num2str(K_(k)/N)); end;
   strM_ = []; for m=1:length(M_), strM_ = strvcat(strM_,num2str(M_(m)/2/N)); end;
 else % plot K and M on axes
   strK_ = []; for k=1:length(K_), strK_ = strvcat(strK_,num2str(K_(k))); end;
   strM_ = []; for m=1:length(M_), strM_ = strvcat(strM_,num2str(M_(m))); end;
 end
 if time_wall<inf, tit_str1 = [tit_str1,', wall=',num2str(time_wall)]; end;

 % plot results
 if (length(K_)==1)||(length(M_)==1), % non-contour plots...

   if (length(K_)>1) % plot versus K_
     if normalized_axes
       horizontal = K_/N;
       x_str = 'sparsity rate K/N';
     else
       horizontal = K_;
       x_str = 'signal sparsity K';
     end
     if exist('strsplit')
       tit_str1_split = strsplit(tit_str1,'snr=');
       tit_str1 = [tit_str1_split{1},'M=',strM_,', snr=',tit_str1_split{2}];
     end
   elseif (length(M_)>1) % plot versus M_
     if normalized_axes
       horizontal = M_/2/N;
       x_str = 'measurement rate M/(2N)';
     else
       horizontal = M_;
       x_str = 'measurements M';
     end
     if exist('strsplit')
       tit_str1_split = strsplit(tit_str1,'snr=');
       tit_str1 = [tit_str1_split{1},'K=',strK_,', snr=',tit_str1_split{2}];
     end
   end
   if (length(K_)>1)||(length(M_)>1), % plot versus K_ or M_
     figure(1); clf;
       handy = plot(horizontal,success_avg,'.-');
       if exist('setcolor')==2, setcolor(handy,alg_); end;
       axe=axis;
       axis([axe(1:2),-0.1,1.1]);
       legend(alg_,'Location','Best');
       title(['success@',num2str(success_dB),'dB, ',tit_str1])
       xlabel(x_str)
       ylabel('empirical probability')
       grid on
     figure(2); clf;
       handy = semilogy(horizontal,time_ran_out,'.-');
       if exist('setcolor')==2, setcolor(handy,alg_); end;
       legend(alg_,'Location','Best');
       if outage>0,
         title([num2str(100*outage,3),'pct-runtime, ',tit_str1])
       else
         title(['avg-runtime, ',tit_str1])
       end;
       xlabel(x_str)
       ylabel('seconds')
       grid on
     figure(3); clf;
       handy = plot(horizontal,squeeze(nrse_out_dB),'.-');
       if exist('setcolor')==2, setcolor(handy,alg_); end;
       legend(alg_,'Location','Best');
       if outage>0,
         title([num2str(100*outage,3),'pct-NRSE, ',tit_str1])
       else
         title(['avg-NRSE, ',tit_str1])
       end;
       xlabel(x_str)
       ylabel('dB')
       grid on
     figure(4); clf;
       handy = plot(horizontal,squeeze(nmse_out_dB),'.-');
       if exist('setcolor')==2, setcolor(handy,alg_); end;
       legend(alg_,'Location','Best');
       if outage>0,
         title([num2str(100*outage,3),'pct-NMSE-BC, ',tit_str1])
       else
         title(['avg-NMSE-BC, ',tit_str1])
       end;
       xlabel(x_str)
       ylabel('dB')
       grid on
     figure(5); clf;
       handy = semilogy(horizontal,squeeze(trys_out),'.-');
       if exist('setcolor')==2, setcolor(handy,alg_); end;
       legend(alg_,'Location','Best');
       if outage>0,
         title([num2str(100*outage,3),'pct-tries, ',tit_str1])
       else
         title(['avg-tries, ',tit_str1])
       end;
       xlabel(x_str)
       ylabel('tries')
       grid on
   else % print results to screen
     success_avg
     time_ran_out
     nrse_out_dB
     nmse_out_dB
     trys_out
   end;

 else % contour plots...

   if normalized_axes
     y_str = 'sparsity rate K/N';
     x_str = 'measurement rate M/(2N)';
   else
     y_str = 'signal sparsity K';
     x_str = 'number of measurements M';
   end
   
   if 0 % optionally add a synthetic PTC
     alg_ = strvcat(alg_,'M=2K'); % add a new algorithm
     a = size(alg_,1); % index of new algorithm
     for k=1:length(K_),
       for m=1:length(M_),
         if (M_(m) >= 2*K_(k))
           success_avg(k,m,a) = 1;
         else
           success_avg(k,m,a) = 0;
         end
       end
     end;
     time_ran_out(:,:,a) = 1;
     nrse_out_dB(:,:,a) = rand(size(nrse_out_dB(:,:,1)));
     nmse_out_dB(:,:,a) = rand(size(nmse_out_dB(:,:,1)));
     trys_out(:,:,a) = 1;
   end

   % extract matlab contour-based PTC
   M_ptc = cell(1,size(alg_,1));
   for a=1:size(alg_,1),
    cs = contourc(success_avg(:,:,a),ptc*[1 1]);
    if isempty(cs)
      display([alg_(a,:),': success contour is empty!'])
    else
      M_ptc{a} = cs;
    end;
   end;%for a

   % plot either contour or imagesc results
   if strcmp(plot_type,'imagesc')||strcmp(plot_type,'ptc'),
     color = 'gray';
     if 0 % remove non power-of-2 axis labels
       for k=1:size(strK_,1), 
         if 2^floor(log2(K_(k)))~=K_(k), 
	   for i=1:size(strK_,2), strK_(k,i) = ' '; end; 
	 end; 
       end;
       for m=1:size(strM_,1), 
         if 2^floor(log2(M_(m)))~=M_(m), 
	   for i=1:size(strM_,2), strM_(m,i) = ' '; end; 
	 end; 
       end;
     end
     for a=1:size(alg_,1),
       figure(5*a-4); clf;
         colormap(color)
         imagesc(success_avg(:,:,a),[0,1]);
         set(gca,'YDir','normal'); 
         set(gca,'YTick',1:length(K_))
         set(gca,'YTickLabel',strK_)
         set(gca,'XTick',1:length(M_))
         set(gca,'XTickLabel',strM_)
         title([alg_(a,:),': success@',num2str(success_dB),'dB, ',tit_str1])
         ylabel(y_str)
         xlabel(x_str)
         handy = colorbar; 
         ylabel(colorbar,'empirical probability')
%ylabel('K')
%xlabel('M')
%if EM, title('EM-P-BiG-AMP'); else title('P-BiG-AMP'); end;
         if strcmp(plot_type,'ptc')
           hold on;
           plot(M_ptc{a}(1,2:end),M_ptc{a}(2,2:end));
	   hold off;
         end;
         pause(eps)
       figure(5*a-3); clf;
         imagesc(time_ran_out(:,:,a),[0,max(max(time_ran_out(:,:,a)))]);
         gray = colormap(color);
         set(gcf,'Colormap',gray(end:-1:1,:));
         set(gca,'YDir','normal'); 
         set(gca,'YTick',1:length(K_))
         set(gca,'YTickLabel',strK_)
         set(gca,'XTick',1:length(M_))
         set(gca,'XTickLabel',strM_)
         if outage>0,
           title([alg_(a,:),': ',num2str(100*outage,3),'pct-runtime, ',tit_str1])
         else
           title([alg_(a,:),': avg-runtime, ',tit_str1])
         end;
         ylabel(y_str)
         xlabel(x_str)
         handy = colorbar; 
         ylabel(colorbar,'seconds')
         pause(eps)
       figure(5*a-2); clf;
         imagesc(nrse_out_dB(:,:,a),...
       		[min(min(nrse_out_dB(:,:,a))),max(max(nrse_out_dB(:,:,a)))]);
         gray = colormap(color);
         set(gcf,'Colormap',gray(end:-1:1,:));
         set(gca,'YDir','normal'); 
         set(gca,'YTick',1:length(K_))
         set(gca,'YTickLabel',strK_)
         set(gca,'XTick',1:length(M_))
         set(gca,'XTickLabel',strM_)
         if outage>0,
           title([alg_(a,:),': ',num2str(100*outage,3),'pct-NRSE, ',tit_str1])
         else
           title([alg_(a,:),': avg-NRSE, ',tit_str1])
         end;
         ylabel(y_str)
         xlabel(x_str)
         handy = colorbar; 
         ylabel(colorbar,'dB')
         pause(eps)
       figure(5*a-1); clf;
         imagesc(nmse_out_dB(:,:,a),...
       		[min(min(nmse_out_dB(:,:,a))),max(max(nmse_out_dB(:,:,a)))]);
         gray = colormap(color);
         set(gcf,'Colormap',gray(end:-1:1,:));
         set(gca,'YDir','normal'); 
         set(gca,'YTick',1:length(K_))
         set(gca,'YTickLabel',strK_)
         set(gca,'XTick',1:length(M_))
         set(gca,'XTickLabel',strM_)
         if outage>0,
           title([alg_(a,:),': ',num2str(100*outage,3),'pct-NMSE, ',tit_str1])
         else
           title([alg_(a,:),': avg-NMSE, ',tit_str1])
         end;
         ylabel(y_str)
         xlabel(x_str)
         handy = colorbar; 
         ylabel(colorbar,'dB')
         pause(eps)
       figure(5*a); clf;
         imagesc(trys_out(:,:,a),...
       		[min(min(trys_out(:,:,a)))-1,max(max(trys_out(:,:,a)))]);
         gray = colormap(color);
         set(gcf,'Colormap',gray(end:-1:1,:));
         set(gca,'YDir','normal'); 
         set(gca,'YTick',1:length(K_))
         set(gca,'YTickLabel',strK_)
         set(gca,'XTick',1:length(M_))
         set(gca,'XTickLabel',strM_)
         if outage>0,
           title([alg_(a,:),': ',num2str(100*outage,3),'pct-tries, ',tit_str1])
         else
           title([alg_(a,:),': avg-tries, ',tit_str1])
         end;
         ylabel(y_str)
         xlabel(x_str)
         handy = colorbar; 
         ylabel(colorbar,'tries')
         pause(eps)
     end;
 
   elseif strcmp(plot_type,'contour'), 
     for a=1:size(alg_,1),
       figure(a); clf;
       colormap(color)
       %levels = [ceil(skip_pct*20)/20:0.025:1];
       levels = [0.25:0.25:0.75];
       [cs,jim] = contour(success_avg(:,:,a),levels);
       set(gca,'YTick',1:length(K_))
       set(gca,'YTickLabel',strK_)
       set(gca,'XTick',1:length(M_))
       set(gca,'XTickLabel',strM_)
       h=clabel(cs,jim); 
       for i=1:length(h), set(h(i),'FontSize',8); end;
       grid on;
       title([alg_(a,:),': success@',num2str(success_dB),'dB, ',tit_str1])
       ylabel(y_str)
       xlabel(x_str)
       pause(eps)
     end;

   else
     error('unsupported plot_type'), 
   end;

   % plot summary ptc
   linetypes = {'*','o','^','x','s','d','v'};
   figure(5*a+1);clf;
     handy = [];
     for a=1:size(alg_,1), 
       handy = [handy;plot(M_ptc{a}(1,2:end),M_ptc{a}(2,2:end))];
       %set(handy(a),'Marker',linetypes{a}); 
       if a==1, hold on; end;
     end;
     if exist('setcolor2')==2, setcolor2(handy,alg_); end;

     hold off;
     axis([1,length(M_),1,length(K_)])
     set(gca,'YTick',1:length(K_))
     set(gca,'YTickLabel',strK_)
     set(gca,'XTick',1:length(M_))
     set(gca,'XTickLabel',strM_)
     legend(alg_,'Location','NorthWest');
     grid on;
     title([num2str(ptc*100),'pct-success@',num2str(success_dB),'dB, ',tit_str1])
     ylabel(y_str)
     xlabel(x_str)
     pause(eps)

   % optionally superimpose a custom trace on a ptc plot
   if 1
     figure(1); 
     hold on
       MM = [0,M_(end)];
       KK = MM/2;
       if strcmp(spaceKM,'lin')
         kk = 1+(length(K_)-1)*(KK-K_(1))/(K_(end)-K_(1));
         mm = 1+(length(M_)-1)*(MM-M_(1))/(M_(end)-M_(1));
       else
         kk = 1+(length(K_)-1)*(log2(KK)-log2(K_(1)))/(log2(K_(end))-log2(K_(1)));
         mm = 1+(length(M_)-1)*(log2(MM)-log2(M_(1)))/(log2(M_(end))-log2(M_(1)));
       end
       handy = [handy; plot(mm,kk,'r-')];
       set(handy(end),'LineWidth',1.0);
       alg_ = strvcat(alg_,'M=2K');
       %legend('M=2K')
     hold off
   end

 end; % if non-contour or contour plots

