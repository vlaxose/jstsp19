function extractFrame

close all
vidlist = [2]; 
%vidlist = [1];
mvid = length(vidlist);

est_rank = 0;   estk = 100; k = 100; rank_max = k;
est_rank = 2;   estk = 5;   k = 50; rank_max = k;
est_rank = 0;   k = 50;     estk = k; rank_max = k;
%est_rank = 0;   k = 30;     estk = k; rank_max = k;

%est_rank = 2;   estk = 50;   k = 100; rank_max = k;
tol = 1e-4; est_rank = 0;   k = 60;     estk = k; rank_max = k;

tol = 1e-4; est_rank = 2;   estk = 50;   k = 120; rank_max = k;
tol = 5e-4; est_rank = 2;   estk = 50;   k = 100; rank_max = k;
tol = 5e-4; est_rank = 2;   estk = 20;   k = 100; rank_max = k;
tol = 1e-4; est_rank = 2;   estk = 50;   k = 100; rank_max = k;

tol = 1e-3; est_rank = 2;   estk = 20;   k = 100; rank_max = k;


tol = 1e-3; est_rank = 2;   estk = 20;   k = 80; rank_max = k;

%tol = 5e-4; est_rank = 2;   estk = 50;   k = 90; rank_max = k;
%tol = 1e-4; est_rank = 0;   estk = 120;   k = 120; rank_max = k;

for di = 1:mvid
    vid = vidlist(di);
    switch vid
        case 1; load rhinos;
        case 2; load xylo;
        otherwise;
            error('vid must be 1 or 2');
    end
end

%load(strcat('res-mov',num2str(vid), 'rank', num2str(k)), 'Mc', 'Mn');



%load(strcat('res-mov',num2str(vid), 'est',num2str(est_rank), 'rank', num2str(estk), 'max', num2str(rank_max)), 'Mc', 'Mn');
%solver = 'lmafit'; name = 'all';

%load(strcat('res-rgb-mov-',num2str(vid), 'est',num2str(est_rank), 'rank', num2str(k), 'max', num2str(rank_max)), 'Mc', 'Mn');
%nsolver = 'lmafit';ame = 'rgb';

load(strcat('APGL-mov',num2str(vid)), 'McA', 'Mn'); Mc = McA'; clear McA;
solver = 'apgl'; name = 'all';

strcat('APGL-mov',num2str(vid))

Mato = mat2rgb(Mo, imsize); clear Mo
Matn = mat2rgb(Mn, imsize); clear Mn
Matc = mat2rgb(Mc, imsize); clear Mc

idx = 1;
fig = figure(1); imshow(Mato(:,:,:,idx));
print(fig , '-depsc',strcat('../result/vid',num2str(vid),name,'o-frame',num2str(idx), 'rank', num2str(k),'.eps'));
fig = figure(2); imshow(Matn(:,:,:,idx));
print(fig , '-depsc',strcat('../result/vid',num2str(vid),name,'n-frame',num2str(idx), 'rank', num2str(k),'.eps'));
fig = figure(3); imshow(Matc(:,:,:,idx));
print(fig , '-depsc',strcat('../result/',solver,'-vid',num2str(vid),name,'c-frame',num2str(idx),'rank', num2str(k),'.eps'));



idx = 50;
fig = figure(1); imshow(Mato(:,:,:,idx));
print(fig , '-depsc',strcat('../result/vid',num2str(vid),name,'o-frame',num2str(idx), 'rank', num2str(k),'.eps'));
fig = figure(2); imshow(Matn(:,:,:,idx));
print(fig , '-depsc',strcat('../result/vid',num2str(vid),name,'n-frame',num2str(idx), 'rank', num2str(k),'.eps'));
fig = figure(3); imshow(Matc(:,:,:,idx));
print(fig , '-depsc',strcat('../result/',solver,'-vid',num2str(vid),name,'c-frame',num2str(idx), 'rank', num2str(k),'.eps'));


idx = 100;
fig = figure(1); imshow(Mato(:,:,:,idx));
print(fig , '-depsc',strcat('../result/vid',num2str(vid),name,'o-frame',num2str(idx), 'rank', num2str(k),'.eps'));
fig = figure(2); imshow(Matn(:,:,:,idx));
print(fig , '-depsc',strcat('../result/vid',num2str(vid),name,'n-frame',num2str(idx), 'rank', num2str(k),'.eps'));
fig = figure(3); imshow(Matc(:,:,:,idx));
print(fig , '-depsc',strcat('../result/',solver,'-vid',num2str(vid),name,'c-frame',num2str(idx), 'rank', num2str(k),'.eps'));

end
