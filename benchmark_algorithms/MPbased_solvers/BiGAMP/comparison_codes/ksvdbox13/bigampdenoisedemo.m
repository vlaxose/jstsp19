function bigampdenoisedemo






%% prompt user for image %%

pathstr = fileparts(which('ksvddenoisedemo'));
dirname = fullfile(pathstr, 'images', '*.png');
imglist = dir(dirname);

imnum = 1;

imgname = fullfile(pathstr, 'images', imglist(imnum).name);



%% generate noisy image %%

sigma = 20;

disp(' ');
disp('Generating noisy image...');

im = imread(imgname);
im = double(im);

n = randn(size(im)) * sigma;
imnoise = im + n;



%% set parameters %%

params.x = imnoise;
params.blocksize = 8;
params.dictsize = 256;
params.sigma = sigma;
params.maxval = 255;
params.trainnum = 40000;
params.iternum = 20;
params.memusage = 'high';



% denoise!
disp('Performing BiG-AMP denoising...');
[imout, dict] = bigampdenoise(params);



% show results %

dictimg = showdict(dict,[1 1]*params.blocksize,round(sqrt(params.dictsize)),round(sqrt(params.dictsize)),'lines','highcontrast');
figure; imshow(imresize(dictimg,2,'nearest'));
title('Trained dictionary');

figure; imshow(im/params.maxval);
title('Original image');

figure; imshow(imnoise/params.maxval); 
title(sprintf('Noisy image, PSNR = %.2fdB', 20*log10(params.maxval * sqrt(numel(im)) / norm(im(:)-imnoise(:))) ));

figure; imshow(imout/params.maxval);
title(sprintf('Denoised image, PSNR: %.2fdB', 20*log10(params.maxval * sqrt(numel(im)) / norm(im(:)-imout(:))) ));

