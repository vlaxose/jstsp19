N = 16;
Nr = 64;
[~,Zbar_2] = wideband_mmwave_channel(4, Nr, N, 2, 3, Nr, N);
[~,Zbar_4] = wideband_mmwave_channel(8, Nr, N, 2, 3, Nr, N);
[~,Zbar_8] = wideband_mmwave_channel(12, Nr, N, 2, 3, Nr, N);

figure;
subplot(1,3,1)
bar3(abs(Zbar_4));title('L=4')
subplot(1,3,2)
bar3(abs(Zbar_8));title('L=8')
subplot(1,3,3)
bar3(abs(Zbar_12));title('L=12')

