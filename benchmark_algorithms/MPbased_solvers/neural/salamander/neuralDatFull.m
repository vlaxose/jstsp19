% neuralDat:  Generates matlab data file for neural processing.
%
% The program, optionally, generates two files:
%
% data/StimFull:  Contains the stimulation matrix A(ibin,ipix) = 1 
% if pixel ipix was stimulated in time bin ibin.  Since the size of this
% matrix would be large it is stored as BinaryMatrix object, StimFull,
% in a packed form.
%
% Also, for each channel, the program generates a file of the form,
% data/sta_%s where %s is the channel string spikeStr.  This file contains
% the STA estimate and spike count cnt.

% Data parameters
ndly = 30;  % number of delays to test
            
% Select files to create.  If an option is set to false, the
% file is loaded.
genStimFull = false;    % generate the StimFull matrix.  
compSta = true;        % generate the STA and spike count
spikeStr = 'ch15_c1';   % channel

% miscellaneous parameters
chi = 6; % electrode ch
ci = 1;  % cell # on chi
filei = 1; % stim file in rdata
msec = 20; % spike clock pts/ms
fr_align = 3; % frame marker alignment ref
xpix = 80; % number of x-checkers
ypix = 60; % number of y-checkers
xres = 8;  % x-checker pixel size
yres = 8;  % y-checker pixel size
klen = 300; %msec legnth for kernel
kbinlen = 10; % bin size for kernel
kbins = klen/kbinlen; % kernel length in bins
imsizeB = xpix*ypix/8; %image size in bytes
Ldim = [xpix ypix kbins xres yres kbinlen 1];


% Load date with spike responses
load data\rfdata_051213_traj3.mat;

% Generate the packed stimulation matrix
if (genStimFull)
    ninTot = xpix*ypix;
    
    % Open file with spike data
    fid = fopen('data\uwn_8x8x2_120600f_a.u1','r','l');

    StimFull = genStimMatrixFull(fid, fmap, ninTot, []); 
    
    save data/neuralStimFull StimFull;
    
    flcose(fid);
else
    
    load data/neuralStimFull;
end

% Generate spike count
if (compSta)
    
    cmd = sprintf('spikes = spikes_%s;', spikeStr);
    eval(cmd);
    
    nspikesTot = length(spikes);
    nbinsTot = StimFull.nrow;
    cnt = zeros(nbinsTot,1);
    
    for is = 1:nspikesTot
       ibin = spikes(is);
       cnt(ibin) = cnt(ibin) + 1;
    end
    
    % Compute the STA
    ndly = 30;
    dispNum = 100;
    staSub = StimFull.firFiltTr(cnt,ndly,[], dispNum);
    staSub = staSub - mean(mean(staSub));
    cmd = sprintf('save data/sta_%s staSub cnt xpix ypix kbinlen', spikeStr);
    eval(cmd);
else
    cmd = sprintf('load data/sta_%s', spikeStr);
    eval(cmd);
end


% Plot the STA estimate
xsq = zeros(xpix,ypix,ndly);
xmin = min(min(staSub));
xmax = max(max(staSub));
for idly = 1:ndly
    xsq(:,:,idly) = reshape(staSub(idly,:),xpix,ypix);
    imagesc(xsq(:,:,idly), [xmin xmax]);
    axis equal;
    pause(.1);        
end


