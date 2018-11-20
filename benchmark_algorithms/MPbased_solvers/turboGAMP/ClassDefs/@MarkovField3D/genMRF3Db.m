function [X] = genMRF3Db(obj, N1, N2, N3, NGibbsIter, verbose)

    % Extract MRF parameters from MarkovField object
    betax = obj.betax;
    betay = obj.betay;
    betaz = obj.betaz;
    lambda = obj.alpha;
    
    % Randomly initialize spin states (+/-1 configuration)
%     X = (rand(N1,N2,N3) < lambda)*2 - 1;
    X = -ones(N1,N2,N3);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    
    for n=1:NGibbsIter
        
        % Pad current X with zeros in all dimensions to facilitate a simple
        % way of computing the net magnetic contribution of neighboring
        % spins at each location in the lattice
        Xpad = padarray(X, [1, 1, 1]);
        
        % Calculate the magnetic contribution of neighbors of each cell
        neighbors = betax * circshift(Xpad, [1, 0, 0]) + ...
            betax * circshift(Xpad, [-1, 0, 0]) + ...
            betay * circshift(Xpad, [0, 1, 0]) + ...
            betay * circshift(Xpad, [0, -1, 0]) + ...
            betaz * circshift(Xpad, [0, 0, 1]) + ...
            betaz * circshift(Xpad, [0, 0, -1]);
        
        % Remove zero-padded edges from neighbors array
        neighbors = neighbors(2:end-1,2:end-1,2:end-1);
        
        % Calculate the change in energy of flipping a spin
        DeltaE = (X .* (neighbors - lambda));
        
        % Calculate the transition probabilities
        p_trans = exp(-DeltaE);
        
%         p_trans = p_trans(:);
%         psrt = sort(p_trans(X(:) == 1), 'descend');
%         pcut = psrt(round(lambda*N1*N2*N3));
%         p_trans(X(:) == 1) = (p_trans(X(:) == 1) >= pcut);
%         psrt = sort(p_trans(X(:) == -1), 'descend');
%         pcut = psrt(round(lambda*N1*N2*N3));
%         p_trans(X(:) == -1) = (p_trans(X(:) == -1) >= pcut);
%         p_trans = reshape(p_trans, [N1, N2, N3]);
        
        % Decide which transitions will occur
        transitions = (rand(N1,N2,N3) < p_trans ).*(rand(N1,N2,N3) < 0.10)*-2 + 1;
        
        % Perform the transitions
        X = X .* transitions;
        
        if verbose
            fprintf('Transition %%: %g\n', sum(transitions(:) == -1) / N1 / N2 / N3 * 100);
            fprintf('Sparsity %%: %g\n', sum(X(:) == 1) / N1 / N2 / N3 * 100);
        end
        
    end
    
    % Convert +/-1 representation to 0-1 representation
    X = (X > 0);    
    
    % Vectorize
    X = X(:);
end
