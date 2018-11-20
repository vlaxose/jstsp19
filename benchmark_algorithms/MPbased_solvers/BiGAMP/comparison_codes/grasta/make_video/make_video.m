function make_video( video_name, video_matrix_fg,video_matrix_bg, video_matrix,vInfo, vTitle ,RESIZE)
%MAKE_VIDEO Summary of this function goes here
%   Detailed explanation goes here
if nargin < 7, RESIZE = 1; end

frame_count = size(video_matrix_fg,2);
rows = vInfo.rows; cols = vInfo.cols;
vidObj = VideoWriter(video_name);
open(vidObj);

figure(1); set(gcf,'position',[100,400,400+3*cols,100+rows]);
hfg = subplot(1,3,1);set(gca,'nextplot','replacechildren'); title([vTitle ' FG']);
hbg = subplot(1,3,2);set(gca,'nextplot','replacechildren'); title([vTitle ' BG']);
hvideo = subplot(1,3,3);set(gca,'nextplot','replacechildren'); title('Original');

colormap gray;axis off;
clims = [-0.8 0.8];
for i=1:frame_count,
    
    img = reshape(video_matrix_fg(:,i),rows/RESIZE,cols/RESIZE);
%     mmax = max(abs(img(:))); 
%     if  mmax < 0.2,
%         img(1,1) = 1;
%     end
    
    axes(hfg); imagesc(abs(img),clims); colormap gray;axis off; axis ij ; 
    
    img = reshape(video_matrix_bg(:,i),rows/RESIZE,cols/RESIZE);
    axes(hbg); imagesc((img)); colormap gray;axis off; axis ij ; 
    
    img = reshape(video_matrix(:,i),rows,cols);
    axes(hvideo); imagesc((img)); colormap gray;axis off; axis ij ; 

    % Write each frame to the file.
    currFrame = getframe(gcf);
    writeVideo(vidObj,currFrame);
end

% Close the file
close(vidObj);
fprintf('Video have been written into %s video file\n',video_name);


end

