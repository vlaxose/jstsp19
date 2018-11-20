function plotUtilityNew(results,ylimits,fignum1,fignum2,filename1,filename2)

%Utility for quickly printing results


figure(fignum1)
clf
lvals = {};

%Plot the errors
for kk = 1:length(results)
    plot(results{kk}.errHist,getLineStyle(results{kk}.name))
    hold on
    lvals{end+1} = results{kk}.name; %#ok<*AGROW>
end

%Labels
xlabel('iteration')
ylabel('Z NMSE (dB)')
legend(lvals)
ylim(ylimits)
grid

%Save it if directed
if nargin > 5
    print('-depsc',filename1);
end

%Plot timing information vs. performance
figure(fignum2)
clf

%Plot the errors
for kk = 1:length(results)
    plot(results{kk}.timeHist,...
        results{kk}.errHist,getLineStyle(results{kk}.name))
    hold on
end

%Labels
xlabel('time (sec)')
ylabel('Z NMSE (dB)')
legend(lvals)
ylim(ylimits)
grid


%Save it if directed
if nargin > 5
    print('-depsc',filename2);
end


