function extractFrame

close all
vidlist = [2]; 
% vidlist = [1];
mvid = length(vidlist);

k = 50;
%k = 100;
for di = 1:mvid
    vid = vidlist(di);
    switch vid
        case 1; load rhinos_idx;
        case 2; load xylo_idx;
        otherwise;
            error('vid must be 1 or 2');
    end
end

load(strcat('lmafit-mov-idx-',num2str(vid), 'rank', num2str(k)), 'Mc', 'Mn');

idx = 100;
fig = figure(1); image(reshape(Moi(:,idx), imsize)); colormap(cmap);
fig = figure(2); image(reshape(Mn(:,idx), imsize)); colormap(cmap);
fig = figure(3); image(reshape(Mc(:,idx), imsize)); colormap(cmap);

end
