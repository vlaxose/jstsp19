%% This function is to seperate the given video frames into foreground/background 
% in realtime by median filter, and return the corresponding 
% seperated results of ground-truths as fgs_cmp.
%
%%

function [fgs_cmp,bgs_cmp,vInfo] = bgfg_seperation_mf(videopath, MEDIAN_BUF, thresh,FPS_ONLY, frame_names,MAX_FRAME )

FILE_EXT        = '.bmp';

% Median-filter seperation
%
Dir_lists = dir(videopath);
Video_Length = length(Dir_lists);

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
    
    if frame_count==1,
        [rows,cols]     = size(I);
        VW_ROWS         = rows;
        VW_COLS         = cols; %ceil(cols * VW_RATIO);
        DIM             = VW_ROWS * VW_COLS;
        bgbuf           = zeros(DIM, MEDIAN_BUF);
        k               = 1;
        fgs_cmp         = zeros(DIM, length(frame_names));
        bgs_cmp         = zeros(DIM, length(frame_names));
        vInfo.rows = rows;
        vInfo.cols = cols;
        
    end
    
    I = I/max(max(I));
        
    [ fg, bg, bgbuf, k ] = median_filter_bg( I(:), bgbuf, MEDIAN_BUF, k );
    
    noise_thresh = 1 * min(abs(I(:)));
    fg(abs(fg) < noise_thresh) = 0;

    bg_img = reshape(bg , VW_ROWS,VW_COLS);
    o_img = reshape( I ,VW_ROWS,VW_COLS );
    s_img = reshape(fg,VW_ROWS,VW_COLS);
    fg = fg_thresholding(fg,thresh);
    s_img_bw = reshape(fg,VW_ROWS,VW_COLS);
    
    for jj=1:length(frame_names),
        if strcmpi(Dir_lists(i).name, frame_names{jj}),
            fgs_cmp(:,jj) = s_img(:);
            bgs_cmp(:,jj) = bg_img(:);
            break;
        end
    end

    if ~FPS_ONLY && mod(i,1)==0,
        axes(h_bg); imagesc(bg_img);colormap gray;axis off;axis ij ;
        axes(h_img); imagesc(o_img);colormap gray;axis off;axis ij ;
        axes(h_fg); imagesc(s_img);colormap gray;axis off;axis ij ;        
        axes(h_fg_bw); imagesc(s_img_bw);colormap gray;axis off;axis ij ;        
    end
    
    if frame_count >= MAX_FRAME && MAX_FRAME~=-1,
        break;
    end    
end
t_end = toc(t_start);
fprintf('Seperating %d frames by Median-filter costs %.2f seconds, %.2f fps\n',...
    frame_count, t_end, frame_count/t_end);

end

