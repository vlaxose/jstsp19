function [ frame_names ] = videoFrames( DATASET )
%VIDEOFRAMES Summary of this function goes here
%   Detailed explanation goes here

% DATASET is a string


if strcmp(DATASET,'lobby'),
    beginFrame = 1901; endFrame  = 2500;
    frames_vec = beginFrame:endFrame;
    frame_names = cell(1,length(frames_vec));
    
    for i=1:length(frames_vec)
        frame_names{i} = ['SwitchLight' num2str(frames_vec(i)) '.bmp'];
    end
    
elseif strcmp(DATASET,'hall'),
    beginFrame = 1000; endFrame  = 1600;
    frames_vec = beginFrame:endFrame;
    frame_names = cell(1,length(frames_vec));

    for i=1:length(frames_vec)
        frame_names{i} = ['airport' num2str(frames_vec(i)) '.bmp'];
    end
    
elseif strcmp(DATASET, 'bootstrap'),
    beginFrame = 1000; endFrame  = 1600;
    frames_vec = beginFrame:endFrame;
    frame_names = cell(1,length(frames_vec));

    for i=1:length(frames_vec)
        frame_names{i} = ['b0' num2str(frames_vec(i)) '.bmp'];
    end
    
elseif strcmp(DATASET, 'escalator'),
    beginFrame = 1600; endFrame  = 2200;
    frames_vec = beginFrame:endFrame;
    frame_names = cell(1,length(frames_vec));

    for i=1:length(frames_vec)
        frame_names{i} = ['airport' num2str(frames_vec(i)) '.bmp'];
    end           
    
elseif strcmp(DATASET, 'campus'),   
    beginFrame = 1100; endFrame  = 1700;
    frames_vec = beginFrame:endFrame;
    frame_names = cell(1,length(frames_vec));

    for i=1:length(frames_vec)
        frame_names{i} = ['trees' num2str(frames_vec(i)) '.bmp'];
    end    
    
    
    
elseif strcmp(DATASET, 'curtain'),
    beginFrame = 22600; endFrame  = 23400;
    frames_vec = beginFrame:endFrame;
    frame_names = cell(1,length(frames_vec));

    for i=1:length(frames_vec)
        frame_names{i} = ['Curtain' num2str(frames_vec(i)) '.bmp'];
    end    
    
    
    
elseif strcmp(DATASET, 'fountain'),
    beginFrame = 1000; endFrame  = 1522;
    frames_vec = beginFrame:endFrame;
    frame_names = cell(1,length(frames_vec));
    
    for i=1:length(frames_vec)
        frame_names{i} = ['Fountain' num2str(frames_vec(i)) '.bmp'];
    end    
    
    
    
elseif strcmp(DATASET, 'watersurface'),    
    beginFrame = 1232; endFrame  = 1631;
    frames_vec = beginFrame:endFrame;
    frame_names = cell(1,length(frames_vec));

    for i=1:length(frames_vec)
        frame_names{i} = ['WaterSurface' num2str(frames_vec(i)) '.bmp'];
    end    
    
    
elseif strcmp(DATASET, 'shopping'),
    beginFrame = 1001; endFrame  = 1600;
    frames_vec = beginFrame:endFrame;
    frame_names = cell(1,length(frames_vec));

    for i=1:length(frames_vec)
        frame_names{i} = ['ShoppingMall' num2str(frames_vec(i)) '.bmp'];
    end    
    
end



end

