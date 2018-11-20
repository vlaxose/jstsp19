classdef BiGAMPProblem
    % Problem setup for BiG-AMP. Stores a description of the observed data
    % locations and dimensions
    
    properties
        
        %Problem dimensions. While these parameters may be computable in
        %some cases, for simplicity/generality, BiG-AMP requires that they
        %be set explicitly. The observation matrix is MxL, and N is the
        %inner dimension of the matrix mulitply, e.g. the rank in a matrix
        %completion problem. The value of N may be overriden by the EM
        %codes when doing rank learning, but it must be provided for
        %low-level BiG-AMP routines
        N = [];
        M = [];
        L = [];
        
        %Vectors containing the row,column locations of observed entries of
        %Z. If these are left empty, BiG-AMP assumes that all entries of Z
        %are observed. They should be vectors of length kk, where kk is the
        %number of observed entries in Z.
        rowLocations = [];
        columnLocations = [];
               
    end
    
    methods
        
        % Constructor with default options
        function prob = BiGAMPProblem()
        end
    end
    
end