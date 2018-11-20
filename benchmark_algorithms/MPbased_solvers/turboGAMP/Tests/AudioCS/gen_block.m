function x = gen_block(N,S,K,mod,seed,isuniform)
% ²úÉúÒ»¸ö¿é½á¹¹µÄÏ¡ÊèÐÅºÅ
%   N       - ÐÅºÅ³¤¶È
%   S       - Ï¡Êè¶È
%   K       - ¿éµÄ¸öÊý
%   mod     - Èô²úÉú¸ßË¹Ëæ»úÊý£¬ÔòÉèÖÃÎª1£»
%             Èô²úÉú+/-1Ëæ»úÊý£¬ÔòÉèÖÃÎª0£»
%             Èô²úÉú+1Ëæ»úÊý£¬ÔòÉèÖÃÎª-1£»
%----------------------------------------------
% Version 2, time: 2010.10.21, by <yuleiwhu>
%   add a new parameter to control if someone want to create a
%   blocked sparse signal with uniformly length for each blocks.
%
%   if isuniform, then output uniform length of blocks of sparse signals,
%   else no.
%----------------------------------------------
%

% S = S+1;
if S>N
    error('Ï¡Êè¶ÈÌ«´ó£¬ÒÑ¾­³¬¹ýÁËÐÅºÅµÄ³¤¶È£¬ÇëÉèÖÃ½ÏÐ¡µÄÏ¡Êè¶È£¡');
end

if nargin < 4
    mod = 1; % Ä¬ÈÏ²úÉú¸ßË¹Ëæ»úÊý
end

if nargin < 5
    isrand = 1;
else
    isrand = 0;
end

if nargin <6
    isuniform = 0;
end

if ~isuniform % if not with uniform length of blocks
    if isrand
        temp = randperm(S);
        J = sort(temp(1:K-1),'ascend');
        Jind = [[J(1:end),S];[1,J(1:end)+1]]; % Ã¿¸ö¿éµÄÎ»ÖÃ±êºÅ
        Jlen = [J(1:end),S]-[1,J(1:end)+1]+1;
        
        Nonzeros = randn(S,1);
        
        nZeros = N-S;
        x = zeros(nZeros,1);
        temp = randperm(nZeros);
        idx = sort(temp(1:K),'ascend');
        
        for i = 1:K
            x = [x(1:idx(i));Nonzeros(Jind(2,i):Jind(1,i));x(idx(i)+1:end)];
            idx = idx + Jlen(i);
        end
    else
        rand('seed',seed);
        temp = randperm(S);
        J = sort(temp(1:K-1),'ascend');
        Jind = [[J(1:end),S];[1,J(1:end)+1]]; % Ã¿¸ö¿éµÄÎ»ÖÃ±êºÅ
        Jlen = [J(1:end),S]-[1,J(1:end)+1]+1;
        
        randn('seed',seed);
        Nonzeros = randn(S,1);
        
        nZeros = N-S;
        x = zeros(nZeros,1);
        rand('seed',seed+1000);
        temp = randperm(nZeros);
        idx = sort(temp(1:K),'ascend');
        
        for i = 1:K
            x = [x(1:idx(i));Nonzeros(Jind(2,i):Jind(1,i));x(idx(i)+1:end)];
            idx = idx + Jlen(i);
        end
    end
    
    
    if isequal(mod,0)
        x = sign(x);
    elseif isequal(mod,-1)
        x = abs(x);
    end
    
else % if with uniform length of blocks
    % Generate a uniform length for each blocks, probabilistically.
    prob = ones(K,1)/K;
    bnum = mnrnd(S,prob);
    
    % Generate a uniform distributed spaces, probabilistically.
    prob = ones(K+1,1)/(K+1);
    num = mnrnd(N-S,prob);
    
%     x = zeros(N,1);
    x = zeros(num(1),1);
    for i = 1:K
        x = [x;random('norm',0,1,bnum(i),1);zeros(num(i+1),1)];
    end
    
    if isequal(mod,0)
        x = sign(x);
    elseif isequal(mod,-1)
        x = abs(x);
    end
end
