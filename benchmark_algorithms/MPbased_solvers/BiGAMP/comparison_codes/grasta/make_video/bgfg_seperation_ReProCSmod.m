%% Disclaimer: This function is based on ReProCS(modCS) algorithm downloaded from the authors¡¦ website
% and use the default parameters provided in the demo code of ReProCS to do video background separation.
% In order to run it on our experimental laptop, we resized all the videos for ReProCS by the factor RESIZE
% ROWS = rows/RESIZE; COLS = cols/RESIZE
%
%
% Detailed information on ReProCS please referr to
% http://home.engineering.iastate.edu/ ?chenlu/ReProCS/ReProCS_main.htm
%%

function [fgs_cmp,bgs_cmp, vInfo] = bgfg_seperation_ReProCSmod( videopath,batch_size, MAXITER, thresh,FPS_ONLY, frame_names ,MAX_FRAME,RESIZE)

FILE_EXT        = '.bmp';

Dir_lists = dir(videopath);
Video_Length = length(Dir_lists);


%% 1. use rpca get the clean training data batch_size frames
frame_count = 0;
frames_idx = randperm(Video_Length);

for i=1:Video_Length,
    if Dir_lists(frames_idx(i)).isdir,
        continue;
    end
    if isempty(strfind(Dir_lists(frames_idx(i)).name,FILE_EXT)) ,
        continue;
    end
    
    frame_count = frame_count+1;
    fname = [videopath Dir_lists(frames_idx(i)).name];
    
    % prepare the image
    I = imread(fname);
    I = double(rgb2gray(I));
    
    if frame_count==1,
        [rows,cols]     = size(I);
        VW_ROWS         = rows/RESIZE;
        VW_COLS         = cols/RESIZE; %ceil(cols * VW_RATIO);
        DIM             = VW_ROWS * VW_COLS;
        batch_buf       = zeros(DIM, batch_size);
        RPCA_lambda     = 1/sqrt(DIM);
        fgs_cmp         = zeros(DIM, length(frame_names));  % used for the comparision frames      
        bgs_cmp         = zeros(DIM, length(frame_names));
        batch_fnames    = cell(1,length(frame_names));
        vInfo.rows = rows;
        vInfo.cols = cols;
        
    end
    
    I = imresize(I,[VW_ROWS,VW_COLS],'bicubic');
%     I = I/max(max(I));
    
    k = mod(frame_count, batch_size);    
    if k == 0, k = batch_size; end
    
    batch_buf(:,k) = I(:);
    
    batch_fnames{k} = Dir_lists(i).name;
    
    if mod(frame_count,batch_size) == 0,
        fprintf('randomly selected %d frames for offline rpca\n',batch_size);
        break;
    end
end
t_start = tic;
[BK_hat, ~ ,~] = inexact_alm_rpca(batch_buf, RPCA_lambda, -1, MAXITER);
t_end = toc(t_start);
fprintf('Use %d frames for offline Robust PCA to get the clean training data costs %.2f seconds, %.2f fps\n',...
    batch_size,t_end, frame_count/t_end);

%% 2. training the initial subspace for ReProCSmod
mu0 = mean(BK_hat,2);
numTrain=size(BK_hat,2);
[U, Sig, ~] = svd(BK_hat - repmat(mu0,1,numTrain),0);
T0 = find(diag(Sig)>0.1);
U0 = U(:,T0);Sig0 = Sig(T0,T0);

global tau D d sig_add sig_del % parameters for recursive PCA
D=[]; d=0;
tau=20;
sig_del = 1; sig_add = 1; % thresholds used to update the PC matrix (Ut)

Shat_mod = zeros(DIM,1); Lhat_mod = zeros(DIM,1); Ohat_mod = zeros(DIM,1); Nhat_mod=cell(1,1);
Lhat_mod_old = zeros(DIM,1);
alpha_add = 5;
alpha_del = 10;

Ut = U0; Sigt = Sig0;
clear opts;  opts.tol = 1e-3; opts.print = 0;D=[]; d=0;
opts.delta= 0.05;
gamma = 5;


%% 3. Online RPCA on each frame
if ~FPS_ONLY,
    figure;
    h_fg = subplot(2,2,1);set(gca,'nextplot','replacechildren');title('Foreground');
    h_fg_bw = subplot(2,2,2);set(gca,'nextplot','replacechildren');title('Thresholded-Foreground');
    h_bg = subplot(2,2,3);set(gca,'nextplot','replacechildren');title('Background');
    h_img = subplot(2,2,4);set(gca,'nextplot','replacechildren');title('Video');
end


frame_count = 0;
t_start = tic;
for i=1:Video_Length,
    if Dir_lists(i).isdir,
        continue;
    end
    if isempty(strfind(Dir_lists(i).name,FILE_EXT)) ,
        continue;
    end
    
    frame_count = frame_count+1;
    fname = [videopath Dir_lists(i).name];
    
    % prepare the image
    I = imread(fname);
    I = double(rgb2gray(I));
    %     I = I/max(max(I));
    I = imresize(I,[VW_ROWS,VW_COLS],'bicubic');
    
    if frame_count == 1,
        Atf.times = @(x) Projection(Ut,x); Atf.trans = @(y) Projection(Ut,y);
        yt = Atf.times(I(:)-mu0);
        [xp,~] = yall1(Atf, yt, opts); % xp = argmin ||x||_1 subject to ||yt - At*x||_2 <= opts.delta
        
        %support thresholding and least square
        That = find(abs(xp)>=gamma);
        Shat_mod(That) = subLS(Ut,That,yt);
        Lhat_mod(:) = I(:) - Shat_mod(:);
        
        %estimate Ot
        Ohat_mod(That) = I(That);
        Nhat_mod=That;
               
    else
        Atf.times = @(x) Projection(Ut,x); Atf.trans = @(y) Projection(Ut,y);
        yt = Atf.times(I(:)-mu0);
        Tpred = Nhat_mod;  %predicted support (previous support estimation)
        weights= ones(DIM,1); weights(Tpred)=0;
        opts.weights = weights(:);
        opts.delta = norm(Atf.times(Lhat_mod_old-mu0),2);
        % xp = argmin ||x_{Tpred^c}||_1 subject to ||yt - At*x||_2 <= opts.delta
        [xp,flag] = yall1(Atf, yt, opts);
        
        % Add-LS-Del-LS step
        That = union(Nhat_mod,find(abs(xp)>alpha_add));
        Shat_mod(That) = subLS(Ut,That,yt);
        Tdel = find(abs(Shat_mod(:))<alpha_del); That = setdiff(That,Tdel);
        Shat_mod(That) = subLS(Ut,That,yt); Shat_mod(Tdel) = 0;
        
        % estimate Lt and Ot
        Ohat_mod(That) = I(That);
        Lhat_mod(:) = I(:) - Shat_mod(:);
        Nhat_mod = That;
        
        Lhat_mod_old = Lhat_mod;
        % recursive PCA
        [Ut,Sigt,~]= recursivePCA(Lhat_mod,Ut,Sigt);        
    end
    
    
    bg_img = reshape(Lhat_mod, VW_ROWS,VW_COLS);
    o_img = reshape( I, VW_ROWS,VW_COLS );
    s_img = reshape(Shat_mod, VW_ROWS,VW_COLS);
    fg = fg_thresholding(Shat_mod,thresh);
    s_img_bw = reshape(fg,VW_ROWS,VW_COLS);
    
    for jj=1:length(frame_names),
        if strcmpi(Dir_lists(i).name, frame_names{jj}),
            fgs_cmp(:,jj) = s_img(:);
            bgs_cmp(:,jj) = bg_img(:);
            break;
        end
    end
    
    if ~FPS_ONLY,
        axes(h_bg); imagesc(bg_img);colormap gray;axis off;axis ij ;
        axes(h_img); imagesc(o_img);colormap gray;axis off;axis ij ;
        axes(h_fg); imagesc(s_img);colormap gray;axis off;axis ij ;
        axes(h_fg_bw); imagesc(s_img_bw);colormap gray;axis off;axis ij ;
    end
    
    if mod(frame_count,200)==0,
        fprintf('Processing %d, total %d frames\n',frame_count, Video_Length);
    end
    
    if frame_count > MAX_FRAME && MAX_FRAME > 0,
        break;
    end
end
        
t_end = toc(t_start);
fprintf('Seperating %d frames by ReProCS costs %.2f seconds, %.2f fps\n',...
    frame_count, t_end, frame_count/t_end);


end

