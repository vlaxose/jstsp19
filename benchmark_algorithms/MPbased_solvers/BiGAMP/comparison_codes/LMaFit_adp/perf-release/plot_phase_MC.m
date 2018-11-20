function plot_phase_MC

% load result/lmafit_phase.mat
% 
% [rlist SRlist] = meshgrid(rlist(2:end),SRlist);
% 
% fig = figure(1);
% pcolor(rlist,SRlist, stat(2:end,:)'); 
% shading interp
% colormap (1-gray); 
% colorbar;
% ylabel('sampling ratio'); xlabel('rank'); 
% print(fig , '-deps2','./result/lmafit-pp-1.eps');
% 
% return



load result/lmafit_phase.mat

fig = figure(1);
imagesc(rlist(2:end),SRlist, stat(2:end,:)'); 
shading interp
colormap gray; 
colorbar;
ylabel('sampling ratio','fontsize',14); xlabel('rank','fontsize',14); 
set(gca,'FontSize',14)
print(fig , '-deps2','./result/lmafit-pp-1.eps');


return; %comment out this line to continue

load result/APGL_phase.mat

fig = figure(2);
imagesc(rlist(2:end),SRlist, stat(2:end,:)'); 
shading interp
colormap gray; 
colorbar;
ylabel('sampling ratio','fontsize',14); xlabel('rank','fontsize',14); 
set(gca,'FontSize',14)
print(fig , '-deps2','./result/APGL-pp.eps');


load result/lmafit_FPCA.mat

fig = figure(3);
imagesc(rlist(2:end),SRlist, stat(2:end,:)'); 
shading interp
colormap gray; 
colorbar;
ylabel('sampling ratio','fontsize',14); xlabel('rank','fontsize',14); 
set(gca,'FontSize',14)
print(fig , '-deps2','./result/FPCA-pp.eps');

