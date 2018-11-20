function x=ifft2c_mri(X)
x=fftshift(fft(fftshift(X,1),[],1),1)/sqrt(size(X,1));
x=fftshift(fft(fftshift(x,2),[],2),2)/sqrt(size(X,2));
