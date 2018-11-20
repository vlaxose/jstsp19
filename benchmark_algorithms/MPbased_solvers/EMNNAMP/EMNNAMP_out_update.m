%***********NOTE************
% The user does not need to call this function directly.  It is called
% insead by EMNNAMP.
%
% This function returns real EM updates of the output likelihood's 
% parameters
%
% Coded by: Jeremy Vila, The Ohio State Univ.
% E-mail: vilaj@ece.osu.edu
% Last change: 10/13/14
% Change summary: 
%   v 1.0 (JV)- First release
%   v 2.0 (JV)- Updated output channel parameter learning
%   v 2.1 (JV)- Added option to learn outputs based on hidden Z or X
%               variables.
%   v 2.2 (JV)- changed to true noise variance learning via EM.
%
% Version 2.2

%%
function stateFin = EMNNAMP_out_update(Y, Zhat, Zvar, stateFin, optALG, optEM, lastIter)

%Calculate Problem dimensions
[M, T] = size(Zhat);

%Update output channel stateFineters
if ~optALG.laplace_noise

      %Removed 10-13-14
%     if ~lastIter && ~optEM.maxBethe
%         Zvar = 0;
%     end
    
    %learn AWGN variance.  optEM.hiddenZ automatically toggles from Zhat =
    %A.mult(Xhat) and Zhat = estFin.Zhat (similar for Zvar).
    if optEM.learn_noisevar
        muw = stateFin.noise_var;
        if ~optEM.maxBethe
            if strcmp(optEM.outDim,'joint')
                muw = sum(sum(abs(Y-Zhat).^2))/M/T+sum(sum(Zvar))/M/T;
            elseif strcmp(optEM.outDim,'col')
                muw = sum(abs(Y-Zhat).^2,1)/M+sum(Zvar,1)/M;
            elseif strcmp(optEM.outDIm,'row')
                muw = sum(abs(Y-Zhat).^2,2)/T+sum(Zvar,2)/T;
            end
        else
            %update using amximization of free energy.  Note that the z
            %variables here are actually the S variables.
            if strcmp(optEM.outDim,'joint')
                muw = muw.*sum(sum(abs(Zhat).^2))./sum(sum(Zvar));
            elseif strcmp(optEM.outDim,'col')
                muw = muw(1,:).*sum(abs(Zhat).^2)./sum(Zvar);
            elseif strcmp(optEM.outDim,'row')
                muw = muw(:,1).*sum(abs(Zhat).^2,2)./sum(Zvar,2);
            end
        end
        stateFin.noise_var = resize(muw,M,T);
    end
else
    if optEM.learn_outLapRate
        %The form of the update depends on optEM.hiddenZ
        if optEM.hiddenZ
            % To avoid numerical problems (0/0) when evaluating
            % ratios of Gaussian CDFs, impose a firm cap on the
            % maximum value of entries of rvar
            Zvar = max(1e-60,min(Zvar, 700));

            % Begin by computing various constants on which the
            % posterior mean and variance depend
            sig = sqrt(Zvar);                    % Gaussian prod std dev
            muL = Zhat - Y + stateFin.outLapRate.*Zvar;   % Lower integral mean
            muU = Zhat - Y - stateFin.outLapRate.*Zvar;   % Upper integral mean
            muL_over_sig = muL ./ sig;
            muU_over_sig = muU ./ sig;
            cdfL = normcdf(-muL_over_sig);              % Lower cdf
            cdfU = normcdf(muU_over_sig);               % Upper cdf
            cdfRatio = cdfL ./ cdfU;                    % Ratio of lower-to-upper CDFs
            SpecialConstant = exp( (muL.^2 - muU.^2) ./ (2*Zvar) ) .* ...
                cdfRatio;
            NaN_Idx = isnan(SpecialConstant);        	% Indices of trouble constants

            % For the "trouble" constants (those with muL's and muU's
            % that are too large to give accurate numerical answers),
            % we will effectively peg the special constant to be Inf or
            % 0 based on whether muL dominates muU or vice-versa
            SpecialConstant(NaN_Idx & (-muL >= muU)) = Inf;
            SpecialConstant(NaN_Idx & (-muL < muU)) = 0;

            % Compute the ratio normpdf(a)/normcdf(a) for
            % appropriate upper- and lower-integral constants, a
            RatioL = 2/sqrt(2*pi) ./ erfcx(muL_over_sig / sqrt(2));
            RatioU = 2/sqrt(2*pi) ./ erfcx(-muU_over_sig / sqrt(2));

            % Compute mean
            absZminusYMean = (-1 ./ (1 + SpecialConstant.^(-1))) .* ...
                (muL - sig.*RatioL) + (1 ./ (1 + SpecialConstant)) .* ...
                (muU + sig.*RatioU);

            %Compute lam

            if strcmp(optEM.outDim,'joint')
                outLapRate = M*T./sum(sum(absZminusYMean));
            elseif strcmp(optEM.outDim,'row')
                outLapRate = T./sum(absZminusYMean,2);
            elseif strcmp(optEM.outDim,'col')
                outLapRate = M./sum(absZminusYMean,1);
            end

            %Find indeces of NaNs,  this happens when y = Zhat (no
            %thresholding).  In this case, these indeces give nothing of value
            %to EM updates, so keep those values the same as before.
            outLapRate = resize(outLapRate,M,T);
            ind = isnan(outLapRate);
            outLapRate(ind) = stateFin.outLapRate(ind);
            stateFin.outLapRate = outLapRate;
        else
            Zhat = Zhat - Y;
            sqrtZvar = sqrt(Zvar);
            z_over_mup = Zhat ./ sqrtZvar;
            % Compute the ratio normpdf(a)/normcdf(a) for
            % appropriate upper- and lower-integral constants, a
            RatioL = 2/sqrt(2*pi) ./ erfcx(z_over_mup / sqrt(2));
            RatioU = 2/sqrt(2*pi) ./ erfcx(-z_over_mup / sqrt(2));

            abs_expt = normcdf(z_over_mup).*(Zhat + sqrtZvar.*RatioU) ...
                - normcdf(-z_over_mup).*(Zhat - sqrtZvar.*RatioL);

            if strcmp(optEM.outDim,'joint')
                outLapRate = M*T./sum(sum(abs_expt));
            elseif strcmp(optEM.outDim,'row')
                outLapRate = T./sum(abs_expt,2);
            elseif strcmp(optEM.outDim,'col')
                outLapRate = M./sum(abs_expt,1);
            end

            %Find indeces of NaNs,  this happens when y = Zhat (no
            %thresholding).  In this case, these indeces give nothing of value
            %to EM updates, so keep those values the same as before.
            outLapRate = resize(outLapRate,M,T);
            ind = isnan(outLapRate);
            outLapRate(ind) = stateFin.outLapRate(ind);
            stateFin.outLapRate = outLapRate;
        end
        
    end
end

return