function Afull = genStimMatrixFull(fid, fmap, ninTot, Ispike)
% genStimMatrix:  Generates a stimulation matrix
%
% The stimulation matrix is stored as a packed BinaryMatrix
 
% Get dimensions
imsizeB = ninTot/8;            % image size in bytes      

% By default, get all spikes
if (isempty(Ispike))            
    Ispike = (1:ninTot)';
end
nin = length(Ispike);           % num of input neurons 

% Create null binary packed matrix
disp('Creating full Stimulation matrix...');
Afull = BinaryMatrix(fmap,nin);
for ifr = 1:max(fmap)
    
    if (mod(ifr,500) == 0)
        fprintf(1,'%d\n', ifr);
    end        
        
    % Read the frame from file
    fseek(fid, ifr*imsizeB, 'bof');
    x = fread(fid,ninTot,'ubit1');
    x = x(Ispike);
    
    % Add to matrix
    Afull.setRow(ifr, x);
    
end
