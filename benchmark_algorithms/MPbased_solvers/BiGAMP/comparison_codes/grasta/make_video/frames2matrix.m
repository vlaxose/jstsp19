function [ vMatrix,vInfo ] = frames2matrix( videopath, frame_names)
%FRAMES2MATRIX Summary of this function goes here
%   Detailed explanation goes here
FILE_EXT        = '.bmp';

Dir_lists = dir(videopath);
Video_Length = length(Dir_lists);

frame_count = 0;

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
        DIM             = rows * cols;
        vMatrix         = zeros(DIM, length(frame_names));
        vInfo.rows = rows;
        vInfo.cols = cols;
        
    end
        
    for jj=1:length(frame_names),
        if strcmpi(Dir_lists(i).name, frame_names{jj}),
            vMatrix(:,jj) = I(:);            
            break;
        end
    end
      
end

end

