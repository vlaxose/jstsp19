classdef RankOneFitOpt
    %RankOneFitOpt:  Default options for the rankOneFit program.
    
    properties
        traceHist = 1;  % 1 = trace history of all vectors
        nit = 10;   % number of iterations
        vvarInit = 0;   % iniital variance on v estimate
        
        linEst = 0;         % Use linear estimation
        linEstThresh = 10;  % Ratio of yvar/vvar0 to trigger Gaussian estimation
        
        % Genie debug variables.  Only available when true data is passed
        pgenie = 0;  % 1=use genie for p
        qgenie = 0;  % 1=use genie for q
        SEgenie = 0; % 1=use true second-order stats for SE equations
        compTrue = 1;  % 1=compute values for true vectors u0, v0 

        % Renormalize u and v to theoretical value after each iteration
        normu = 1;   
        normv = 1;          
        
        % Minimum variances
        minau = 0.01;
        minav = 0.01;
        
        % expected correlation coefficients
        corruSE = [];
        corrvSE = [];
    end
    
    methods
        function opt = RankOneFitOpt()
        end
    end
    
end

