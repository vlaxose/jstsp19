function X=fft2c_mri(x)
X=fftshift(ifft(fftshift(x,1),[],1),1)*sqrt(size(x,1));
X=fftshift(ifft(fftshift(X,2),[],2),2)*sqrt(size(x,2));
