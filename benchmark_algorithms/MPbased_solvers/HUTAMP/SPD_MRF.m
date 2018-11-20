% -----------------NOTE-------------------
% This function is called internally by HUTAMP.
% 
% MRF_SPD       This function will perform loopy belief propagation
% on the factor graph defined by the two-dimensional lattice of
% binary support indicator variables, each of which (except along
% edges) are 4-connected.
%
% In addition, this function will perform expectation-maximization
% (EM) updates of the MRF model parameters.
%
% SYNTAX:
% [PI_IN, betaH_upd, betaV_upd, alpha_upd] ...
%    = SPD_MRF(PI_OUT, betaH, betaV, alpha, maxIter, optALG)
%
% INPUTS:
% PI_OUT		An N-by-T matrix of incident messages to the S(n,t)
%				variable nodes, each element being a probability in 
%               [0,1]
% betaH         parameter of MRF
% betaV         parameter of MRF
% alpha         parameter of MRF
% maxIter       Maximum number of iterations
% optALG         Structure containing various EM options
%
% OUTPUTS:
% PI_IN 		An N-by-T matrix of outgoing turbo messages from 
%               the S(n,t) variable nodes, each element being a 
%               probability in [0,1]
% betaH         parameter of MRF
% betaV         parameter of MRF
% alpha         parameter of MRF
% S_POST        Posterior activity probabilities, i.e., Pr{S(n,T) = 1 | Y}
%
% Coded by: Jeremy Vila, Subhojit Som, The Ohio State Univ.
% E-mail: vilaj@ece.osu.edu
% Last change: 2/25/15
% Change summary: 
%   v 1.0 (JV)- First release
%
% Version 1.0

function [PI_IN, betaH_upd, betaV_upd, alpha_upd,S_POST] ...
    = SPD_MRF(PI_OUT, betaH, betaV, alpha, maxIter, optALG)

    % Get sizing information
    [N, T] = size(PI_OUT);

    ebp = exp(betaH);
    ebq = exp(betaV);
    ebpi = exp(-betaH);
    ebqi = exp(-betaV);
    e0 = exp(alpha);
    e1 = exp(-alpha);

    % checkerboard pattern indices
    chk = (checkerboard(1, N/2, T/2) > 0);
    [blackIdx_x, blackIdx_y] = find(chk == 0);
    [whiteIdx_x, whiteIdx_y] = find(chk == 1);

%             La_spdec = reshape(La_spdec,[N Len 1]);

%             nodePot = zeros(Len,Len,2);
%             nodePot(:,:,1) = 1./(1+exp(La_spdec)); % prob that node is 0
%             nodePot(:,:,2) = 1-nodePot(:,:,1); % prob that node is 1
    nodePot = zeros(N,T,2);
    nodePot(:,:,1) = 1 - PI_OUT;    % Prob. that node is 0
    nodePot(:,:,2) = PI_OUT;        % Prob. that node is 1

    % Initialize messages
    msgFromRight = 0.5*ones(N,T,2);
    msgFromLeft = 0.5*ones(N,T,2);
    msgFromTop = 0.5*ones(N,T,2);
    msgFromBottom = 0.5*ones(N,T,2);

    prod0 = zeros(N,T);
    prod1 = prod0;

    for iter = 1:maxIter
        if(mod(iter,2) == 1)
            x = blackIdx_x; y = blackIdx_y;
        else
            x = whiteIdx_x; y = whiteIdx_y;
        end
        % Convert row-column indexing into linear indexing
        ind = sub2ind([N, T], x, y);
        ind1 = sub2ind([N, T, 2], x, y, ones(numel(x),1));
        ind2 = sub2ind([N, T, 2], x, y, 2*ones(numel(x),1));

        % update messages from left 
        prod0(:,2:end) = e0*nodePot(:,1:end-1,1) .* ...
            msgFromLeft(:,1:end-1,1) .* msgFromTop(:,1:end-1,1) .* ...
            msgFromBottom(:,1:end-1,1);
        prod1(:,2:end) = e1*nodePot(:,1:end-1,2) .* ...
            msgFromLeft(:,1:end-1,2) .* msgFromTop(:,1:end-1,2) .* ...
            msgFromBottom(:,1:end-1,2);
        p0 = prod0*ebp + prod1*ebpi;
        p1 = prod0*ebpi + prod1*ebp;
        sump0p1 = p0+p1;

        msgFromLeft(ind1) = p0(ind) ./ sump0p1(ind);
        msgFromLeft(ind2) = p1(ind) ./ sump0p1(ind);
        msgFromLeft(:,1,:) = 0.5;

         % update messages from right 
        prod0(:,1:end-1) = e0*nodePot(:,2:end,1) .* ...
            msgFromRight(:,2:end,1) .* msgFromTop(:,2:end,1) .* ...
            msgFromBottom(:,2:end,1);
        prod1(:,1:end-1) = e1*nodePot(:,2:end,2) .* ...
            msgFromRight(:,2:end,2) .* msgFromTop(:,2:end,2) .* ...
            msgFromBottom(:,2:end,2);
        p0 = prod0*ebp + prod1*ebpi;
        p1 = prod0*ebpi + prod1*ebp;
        sump0p1 = p0 + p1;

        msgFromRight(ind1) = p0(ind) ./ sump0p1(ind);
        msgFromRight(ind2) = p1(ind) ./ sump0p1(ind);
        msgFromRight(:,end,:) = 0.5;

        % update messages from top 
        prod0(2:end,:) = e0*nodePot(1:end-1,:,1) .* ...
            msgFromLeft(1:end-1,:,1) .* msgFromTop(1:end-1,:,1) .* ...
            msgFromRight(1:end-1,:,1);
        prod1(2:end,:) = e1*nodePot(1:end-1,:,2) .* ...
            msgFromLeft(1:end-1,:,2) .* msgFromTop(1:end-1,:,2) .* ...
            msgFromRight(1:end-1,:,2);
        p0 = prod0*ebq + prod1*ebqi;
        p1 = prod0*ebqi + prod1*ebq;
        sump0p1 = p0 + p1;

        msgFromTop(ind1) = p0(ind) ./ sump0p1(ind);
        msgFromTop(ind2) = p1(ind) ./ sump0p1(ind);
        msgFromTop(1,:,:) = 0.5;

         % update messages from bottom 
        prod0(1:end-1,:) = e0*nodePot(2:end,:,1) .* ...
            msgFromRight(2:end,:,1) .* msgFromLeft(2:end,:,1) .* ...
            msgFromBottom(2:end,:,1);
        prod1(1:end-1,:) = e1*nodePot(2:end,:,2) .* ...
            msgFromRight(2:end,:,2) .* msgFromLeft(2:end,:,2) .* ...
            msgFromBottom(2:end,:,2);
        p0 = prod0*ebq + prod1*ebqi;
        p1 = prod0*ebqi + prod1*ebq;
        sump0p1 = p0 + p1;

        msgFromBottom(ind1) = p0(ind) ./ sump0p1(ind);
        msgFromBottom(ind2) = p1(ind) ./ sump0p1(ind);
        msgFromBottom(end,:,:) = 0.5;

    end


    % compute extrinsic likelihood, marginal potential and s_hat
    msgProds = msgFromLeft .* msgFromRight .* msgFromTop .* ...
        msgFromBottom;
    msgProds(:,:,1) = msgProds(:,:,1)*e0;
    msgProds(:,:,2) = msgProds(:,:,2)*e1;
    Le_spdec = log(msgProds(:,:,2)./msgProds(:,:,1));
    PI_IN = 1 ./ (1 + exp(-Le_spdec));

    msgProds = msgProds.*nodePot;
    sumMsgProds = sum(msgProds,3);
    S_POST = msgProds(:,:,2) ./ sumMsgProds;    % Pr{S(n,t) = 1 | Y}
    S_HAT = double(S_POST > 1/2);

    %learn MRF parameters
    if optALG.learn_MRF
        % Compute parameter updates (will require MATLAB's Optimization
        % Toolbox).  Currently learns a single value of beta
        options = optimset('GradObj', 'on', 'Algorithm', 'interior-point', ...
            'MaxFunEvals', 20,'Display', 'off');
        lb = [-1; 0];   % Lower bounds [beta; alpha]
        ub = [1; 1];    % Upper bounds [beta; alpha]
        [updates] = fmincon(@pseudoLF, [betaH; alpha], [], [], [], ...
            [], lb, ub, [], options, S_HAT, N, T);

        betaH_upd = updates(1);
        betaV_upd = updates(1);

        alpha_upd = updates(2);  
    else 
        betaH_upd = betaH;
        betaV_upd = betaV;
        alpha_upd = alpha;
    end
    
    return