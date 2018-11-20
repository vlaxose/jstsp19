function A = genStimMatrix(fid, spikes, fmap, dly, sumOnly, ninTot, Ispike)
% genStimMatrix:  Generates a stimulation matrix
%
% 

% Get dimensions
nspikesTot = length(spikes);   % num of spikes
ndly = length(dly);             % num of delays
imsizeB = ninTot/8;     % image size in bytes

% By default, get all spikes
if (isempty(Ispike))            
    Ispike = (1:ninTot)';
end
nin = length(Ispike);           % num of input neurons 

% Initialize array
if (sumOnly)
    A = zeros(nin,ndly);
else
    A = false(nspikesTot,nin,ndly);
end

% Loop over spikes
lastFr = 0;
for is = 1:nspikesTot
    
    if (mod(is,100) == 0)
        fprintf(1,'%d\n', is);
    end
    for idly = 1:ndly
                
        % Get the frame number
        fr = fmap(max(1,spikes(is)-dly(idly)+1));
        
        % Read the frame from file
        if (fr ~= lastFr)
            fseek(fid, fr*imsizeB, 'bof');
            x = fread(fid,ninTot,'ubit1');
            x = x(Ispike);
        end
        
        % Record the frame data
        if (sumOnly)
            A(:,idly) = A(:,idly) + x;
        else
            A(is,:,idly) = (x > 0);
        end
    end
end
