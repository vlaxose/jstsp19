% MaskedImagingDemo:  Demo of phase-retrieval GAMP on an image, using multiple binary masks

 %Handle random seed
 if verLessThan('matlab','7.14')
   defaultStream = RandStream.getDefaultStream;
 else
   defaultStream = RandStream.getGlobalStream;
 end;
 if 1 % new RANDOM trial
   savedState = defaultStream.State;
   save random_state.mat savedState;
 else
   load random_state.mat
 end
 defaultStream.State = savedState;

 % Demo mode
 DEMO = 1;

 % Parameters
 switch DEMO
   case 1 % compressive N=256^2-pixel image recovery from M=N magnitudes using pre-masked DFTs 
    image_type = 1;
    num_masks = 4;  % number of masks [4]
    num_band = 0;   % no post-randomization [0]
    nZ_over_ndft = sqrt(1/num_masks); % Fourier sampling ratio, per dimension  [0.5]
    dft_zp = 1;	    % zero-padding factor for DFT, per dimension [1]
    SNRdB_true = 30; % SNR in dB [30]
    plot_gamp = 0;  % plot residual trajectory for debugging? (slows things down)
   case 2 % compressive N=256^2-pixel image recovery from M=N/2 magnitudes using post-randomized and pre-masked DFTs 
    image_type = 1;
    num_masks = 2;  % number of masks [2]
    num_band = 10;  % number of non-zero diagonals in banded randomization matrix [10]
    nZ_over_ndft = sqrt(0.5/num_masks); % Fourier sampling ratio, per dimension  [0.5]
    dft_zp = 1;	    % zero-padding factor for DFT, per dimension [1]
    SNRdB_true = 30; % SNR in dB [30]
    plot_gamp = 0;  % plot residual trajectory for debugging? (slows things down)
   case 3 % like case 2 but with N=1024^2-pixel images
    image_type = 2;
    num_masks = 2;  % number of masks [2]
    num_band = 10;  % number of non-zero diagonals in banded randomization matrix [10]
    nZ_over_ndft = sqrt(0.5/num_masks); % Fourier sampling ratio, per dimension  [0.5]
    dft_zp = 1;	    % zero-padding factor for DFT, per dimension [1]
    SNRdB_true = 30; % SNR in dB [30]
    plot_gamp = 0;  % plot residual trajectory for debugging? (slows things down)
 end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 % Load image
 switch image_type,
   case 1
     load('Satellite2.mat','-ascii'); 
     x = Satellite2; 
     clear Satellite2;
   case 2
     load('Satellite2.mat','-ascii'); 
     x = Satellite2; 
     clear Satellite2;
     x = imresize(x,[1024,1024]);
   case 3
     load('Satellite2.mat','-ascii'); 
     x = Satellite2(44:256-58, 51:256-41); 
     %x = x(21:84,21:84); % clip to get smaller image
     %x = x(21:84,1:64); % clip to get smaller image
     clear Satellite2;
 end;
 [nX1,nX2] = size(x);
 x = x(:);
 nx = nX1*nX2;
 indx_big = find( abs(x) > 0.1*sqrt(mean(abs(x).^2)) );
 sparseRat = length(indx_big)/nx;	% proportion of big elements
 xmean0 = mean(x(indx_big));		% mean of big elements
 xvar0 = var(x(indx_big));		% variance of big elements

 % Create the binary masks
 if num_masks>1,
   Mask = round(rand(nX1,nX2,num_masks));	% random binary
   Mask(:,:,num_masks) = 1-Mask(:,:,1);	% make last mask the complement of the first
 else
   Mask = ones(nX1,nX2);			% trivial
 end;

 % Create the linear operator
 ndft1 = 2.^ceil(log2(nX1*dft_zp)); 	% DFT size per dimension
 ndft2 = 2.^ceil(log2(nX2*dft_zp));	% DFT size per dimension
 nZ1 = round(nZ_over_ndft*ndft1);	% number of samples per dimension
 nZ2 = round(nZ_over_ndft*ndft2);	% number of samples per dimension
 if (nZ1>ndft1)||(nZ2>ndft2), error('nZ is too big!'); end;
 num_samp = nZ1*nZ2;
 nz = num_samp*num_masks;
 if num_band == 0,			% subsample the DFT outputs
   if 0 % low-frequency sampling
     omega = zeros(ndft1,ndft2);
     omega(1:nZ1,1:nZ2) = 1;
     SampLocs = find(omega==1);
   else % random sampling
     SampLocs = nan(nZ1*nZ2,num_masks);
     for i=1:num_masks
       SampLocs(:,i) = randperm(ndft1*ndft2,nZ1*nZ2);
     end
   end
   A = sampTHzLinTrans(nX1,nX2,ndft1,ndft2,SampLocs,Mask);
 else					% randomize the DFT outputs
   Band = randn(ndft1*ndft2,num_band)+1i*randn(ndft1*ndft2,num_band);
   A = bandTHzLinTrans(nX1,nX2,ndft1,ndft2,num_samp,Band,Mask);
 end;

 % Compute the noise level based on the specified SNR. 
 z = A.mult(x);
 wvar = 10^(-0.1*SNRdB_true)*mean(abs(z).^2);

 % Generate CAWGN noise 
 w = sqrt(wvar/2)*(randn(nz,2)*[1;1i]); 
 y_abs = abs(z+w);
 SNRdB_hat = 20*log10(norm(z)/norm(w));

%return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 % set PR-GAMP options
 clear opt_pr;
opt_pr.SNRdB = SNRdB_true;	% use true SNR for stopping tolerances
opt_pr.SNRdBinit = SNRdB_true;  % initialize EM with true SNR
opt_pr.nitEM = 0; 		% disable EM iterations
 opt_pr.xreal = 1;		% exploit real-valued aspect of image 
 opt_pr.xnonneg = 1;		% exploit non-negativity of image
 opt_pr.sparseRat = sparseRat;	% exploit sparsity of image (if sparseRat<1)
 opt_pr.xmean0 = xmean0; 	% exploit mean of non-zeros (optional)
 opt_pr.xvar0 = xvar0;		% exploit variance of non-zeros (optional)
 opt_pr.verbose = 1;		% print per-attempt residual [dflt=0]
 opt_pr.plot = plot_gamp; 	% plot trajectory? 
opt_pr.maxTry = 10; 
opt_pr.init_gain = 1; 

 % set exceptions to GAMP options in prGAMP.m
 clear opt_gamp;
 opt_gamp.uniformVariance = 1;	% since faster and more robust here [dflt=1]
%opt_gamp.nit = 150;		% no need for many iterations [dflt=100]
 opt_gamp.adaptStep = 0;	% no adaptive stepsize
 opt_gamp.stepMax = 0.35;	% somewhat aggressive max stepsize [dflt=0.25]
 opt_gamp.step = opt_gamp.stepMax; % start at max stepsize [dflt=stepMax]

 % run PR-GAMP
 tstartGAMP = tic;
 [xhat,out_pr] = prGAMP2(y_abs,A,opt_pr,opt_gamp);
 timeGAMP = toc(tstartGAMP);
 %xhat = disambig2Drfft(xhat,x,nX1,nX2); % very slow!
 mseGAMPdB = 20*log10(norm(x*sign(x'*xhat)-xhat)/norm(x));
 resGAMPdB = 20*log10(norm(abs(A.mult(xhat))-y_abs)/norm(y_abs));
 fprintf(1,'\nFinal Performance:  MSE = %5.1f dB, RES = %5.1f dB, time = %5.1f sec\n', [mseGAMPdB,resGAMPdB,timeGAMP]);

 % plot result
 figure(1); clf;
 cmap = colormap('gray');
 scale = max(x(:))/size(cmap,1);
 subplot(221)
   image(reshape(x/scale,nX1,nX2))
   axis('equal');axis('tight'); 
   title('true')
 subplot(222)
   image(reshape(xhat/scale,nX1,nX2))
   colormap('gray')
   axis('equal');axis('tight'); 
   title(['GAMP (',num2str(mseGAMPdB,3),'dB)'])
 subplot(212)
   lower = sqrt(out_pr.xvar0); 
   upper = sqrt(out_pr.xvar0);
   if opt_pr.xnonneg, lower = min(lower,out_pr.xmean0); end;
   errorbar([1:nx],out_pr.xmean0,lower,upper,'g');
   hold on; 
     plot(x)
     plot(xhat,'r--'); 
   hold off;
   grid on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%run the following to compute average performance
%T=100; mse=nan(1,T); time=nan(1,T); for t=1:T, MaskedImagingDemo; mse(t)=mseGAMPdB; time(t)=timeGAMP; end; mean(mse<-27), median(time)
%figure(4); image(reshape(xhat/scale,nX1,nX2)); colormap('gray'); axis('equal');axis('tight');title(['GAMP (',num2str(mseGAMPdB,3),'dB)'])

