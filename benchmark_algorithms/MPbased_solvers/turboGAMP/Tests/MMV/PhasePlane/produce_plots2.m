% PRODUCE_PLOTS2     Produce plots for BGPHASEPLANE
%

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: zinielj@ece.osu.edu
% Last change: 02/21/12
% Change summary: 
%		- Created (02/21/12; JAZ)
% Version 0.2
%

clear; clc

%% Load a results file

uiload

%% Produce the plots

algs = {'Support-Aware Smoother', 'TurboGAMP (no EM)', ...
    'EMturboGAMP', 'Independent GAMP'};    % Algorithms
suffix = {'sks', 'noem', 'turbo', 'naive'};
N_algs = numel(algs);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot of the NMSE for each approach
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Compute the trial-averaged failure rate, MSE, Jaccard index, TNMSE, and 
% runtime statistics for each algorithm
for b = 1:N_beta
    for d = 1:N_delta
        for g = 1:N_algs
            % First make sure that the current algorithm was included in
            % the dataset being plotted
            if exist(['TNMSE_' suffix{g}]) == 1
                % Current algorithm is present in dataset

                % First identify those data points that are not NaNs, which may
                % be different for different algs.
                eval(['inds = find(~isnan(TNMSE_' suffix{g} '(b,d,:)));']);
                
%                 % Compute the ensemble average of the realizations that are
%                 % better than the Qth quantile w.r.t. TNMSE
%                 Qth = 0.95;     % 95th quantile
%                 warning('Quantiling active')
%                 inds2 = eval(['find(TNMSE_' suffix{g} '(b,d,inds) <= ' ...
%                     'quantile(TNMSE_' suffix{g} '(b,d,inds), Qth));']);
%                 inds = inds(inds2);

                % Compute the ensemble average over the non-NaN trials
                eval(['mean_NSER_' suffix{g} '(b,d) = median(NSER_' suffix{g} ...
                    '(b,d,inds), 3);']);
                eval(['mean_TNMSE_' suffix{g} '(b,d) = median(TNMSE_' suffix{g} ...
                    '(b,d,inds), 3);']);
                eval(['mean_Run_' suffix{g} '(b,d) = median(Run_' suffix{g} ...
                    '(b,d,inds), 3);']);

                % Keep track of the # of realizations averaged over
                eval(['N_trials_' suffix{g} '(b,d) = numel(inds);'])
                
            else
                % Current algorithm is missing, so just assign NaN
                eval(['mean_NSER_' suffix{g} '(b,d) = NaN;']);
                eval(['mean_TNMSE_' suffix{g} '(b,d) = NaN;']);
                eval(['mean_Run_' suffix{g} '(b,d) = NaN;']);
                
                % Keep track of the # of realizations averaged over
                eval(['N_trials_' suffix{g} '(b,d) = 0;'])
            end
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot of the NMSE for each approach as image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure(1); clf;

clim = [-30, 0];    % Colorbar limits

for g = 1:N_algs
    subplot(2,N_algs/2,g)
    imagesc(delta, fliplr(beta), eval(['10*log10(mean_TNMSE_' suffix{g} ')']), clim); hold on
    colorbar; set(gca, 'YTickLabel', flipud(get(gca, 'YTickLabel')));   % Flip beta labels
    
    % Add equal-lambda contours to the plot
    lambda_contours = NaN(N_beta,N_delta);
    for b = 1:N_beta
        for d = 1:N_delta
            M_tmp = round(delta(d)*N);	% Number of measurements per timestep
            lambda_contours(b,d) = beta(b)*M_tmp/N;	% E[K] for (beta, delta) pair
        end
    end
    contour_values = [0.10, 0.20, 0.30, 0.50, 0.75];    % Equal lambda contour lines to plot
    [con_out, con_handle] = contour(delta, fliplr(beta), lambda_contours, ...
        contour_values, 'w-'); 
    clabel(con_out, con_handle, 'Color', 'white'); hold off
    
    ylabel('\beta (E[K]/M)     (More active coefficients)   \rightarrow');
    xlabel('\delta (M/N)    (More measurements)   \rightarrow')
    title_string = [algs{g} ' TNMSE [dB]'];
    title(title_string)
end

% set(gca, 'Color', 'None');
% set(gcf, 'Color', 'None');
% set(gcf, 'InvertHardcopy', 'off');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot of the TNMSE difference from support-aware smoother
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure(2); clf;

clim = [0, 25];    % Colorbar limits

for g = 2:N_algs
    subplot(1,N_algs-1,g-1)
    imagesc(delta, fliplr(beta), eval(['10*log10(mean_TNMSE_' suffix{g} ')' ...
        '- 10*log10(mean_TNMSE_' suffix{1} ')']), clim); hold on
    colorbar; set(gca, 'YTickLabel', flipud(get(gca, 'YTickLabel')));   % Flip beta labels
    
    % Add equal-lambda contours to the plot
    lambda_contours = NaN(N_beta,N_delta);
    for b = 1:N_beta
        for d = 1:N_delta
            M_tmp = round(delta(d)*N);	% Number of measurements per timestep
            lambda_contours(b,d) = beta(b)*M_tmp/N;	% E[K] for (beta, delta) pair
        end
    end
    contour_values = [0.10, 0.20, 0.30, 0.50, 0.75];    % Equal lambda contour lines to plot
    [con_out, con_handle] = contour(delta, fliplr(beta), lambda_contours, ...
        contour_values, 'w-'); 
    clabel(con_out, con_handle, 'Color', 'white'); hold off
    
    ylabel('\beta (E[K]/M)     (More active coefficients)   \rightarrow');
    xlabel('\delta (M/N)    (More measurements)   \rightarrow')
    title_string = [algs{g} ' TNMSE [dB] - ' algs{1}  ' TNMSE [dB]'];
    title(title_string)
end
