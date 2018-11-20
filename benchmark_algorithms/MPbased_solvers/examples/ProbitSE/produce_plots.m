% PRODUCE_PLOTS     Produce plots for the probit channel state-evolution
% phase plane test
%

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: ziniel.1@osu.edu
% Last change: 01/10/13
% Change summary: 
%		- Created (01/10/13; JAZ)
% Version 0.1
%

clear; clc

%% Load a results file

uiload

%% Produce the plots

algs = {'Empirical', 'State Evolution'};       % Algorithms
suffix = {'Emp', 'SE'};             % Storage suffixes
algs2plot = [1, 2];                 % Indices (into algs) of algorithms to plot
cmap = 'copper';
% Nc = 10;                            % # of contours
Nc = linspace(0.15, 0.40, 10);
reformat = false;                   % Resize axis fonts for publication?

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Average over independent trials
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

MeanTestErr_Emp = mean(TestErr_Emp, 3);
MeanTrainErr_Emp = mean(TrainErr_Emp, 3);
MeanRuntime_Emp = mean(Run_Emp, 3);
MeanTestErr_SE = TestErr_SE;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%      Plot test set error
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Clear existing figure
figure(1); clf; colormap(cmap)
clim = [0.10, 0.50];       % Colorbar limits

% Plot the test set error
for a = 1:numel(algs2plot)
    subplot(1, 2, a)
    PlotData = eval(sprintf('MeanTestErr_%s', suffix{algs2plot(a)}));
%     % Uncomment to display an image plot
%     imagesc(delta, fliplr(beta), PlotData, clim); colorbar; 
%     set(gca, 'YTickLabel', flipud(get(gca, 'YTickLabel')));   % Flip beta labels
    
    % Uncomment to display a contour plot
    [C1, H1] = contour(delta, fliplr(beta), flipud(PlotData), Nc);
    clabel(C1, H1, 'Color', 'Black', 'fontsize', 18)
    
    % Add labels
    ylabel('\beta (E[K]/M)  (More active features)   \rightarrow');
    xlabel('\delta (M/N)  (More training samples)   \rightarrow')
    title_string = [algs{algs2plot(a)} ' Test Set Error Rate'];
    title(title_string)
end


% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %      Plot training set error
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% % Clear existing figure
% figure(2); clf; colormap(cmap)
% clim = [0, 0.50];       % Colorbar limits
% 
% % Plot the training set error
% for a = 1:numel(algs2plot)
%     subplot(1, 2, a)
%     PlotData = eval(sprintf('MeanTrainErr_%s', suffix{algs2plot(a)}));
%     % Uncomment to display an image plot
%     imagesc(delta, fliplr(beta), PlotData, clim); colorbar;
%     set(gca, 'YTickLabel', flipud(get(gca, 'YTickLabel')));   % Flip beta labels
%     
% %     % Uncomment to display a contour plot
% %     [C1, H1] = contour(delta, fliplr(beta), flipud(PlotData), Nc);
% %     clabel(C1, H1, 'Color', 'Black', 'fontsize', 18)
%     
%     % Add labels
%     ylabel('\beta (E[K]/M)  (More active features)   \rightarrow');
%     xlabel('\delta (M/N)  (More training samples)   \rightarrow')
%     title_string = [algs{algs2plot(a)} ' Training Misclassification Rate'];
%     title(title_string)
% end
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %             Plot Runtime
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% % Clear existing figure
% figure(5); clf; colormap(cmap)
% 
% % Plot the training set error
% for a = 1:numel(algs2plot)
%     subplot(1, 2, a)
%     PlotData = eval(sprintf('MeanRun_%s', suffix{algs2plot(a)}));
%     % Uncomment to display an image plot
%     imagesc(delta, fliplr(beta), PlotData, clim); colorbar;
%     set(gca, 'YTickLabel', flipud(get(gca, 'YTickLabel')));   % Flip beta labels
%     
% %     % Uncomment to display a contour plot
% %     [C1, H1] = contour(delta, fliplr(beta), flipud(PlotData));
% %     clabel(C1, H1, 'Color', 'Black', 'fontsize', 18)
%     
%     % Add labels
%     ylabel('\beta (E[K]/M)  (More active features)   \rightarrow');
%     xlabel('\delta (M/N)  (More training samples)   \rightarrow')
%     title_string = [algs{algs2plot(a)} ' Runtime (s)'];
%     title(title_string)
% end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       Optional size reformat
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fontsize = 16;
if reformat
    props = {'title', 'xlabel', 'ylabel'};
    for i = 1:5
        for j = 1:4
            figure(i); subplot(2,2,j);
            set(gca, 'FontSize', fontsize);
            for k = 1:3
                hdl = get(gca, props{k});
                set(hdl, 'FontSize', fontsize);
            end
        end
    end
end