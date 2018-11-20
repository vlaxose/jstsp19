classdef PBiGAMPProblem
    % Problem setup for P-BiG-AMP
    
    properties
        
        %P-BiG-AMP estimates b (Nb x 1) and c (Nc x 1) from a set of
        %measurements y (M x 1) of a noiseless true signal z (M x 1). While
        %these sizes can be determined in some cases, for
        %generality/simplicity, we require that the user supply them
        %explicitly.
        Nb = [];
        Nc = [];
        M = [];
        
        %P-BiG-AMP also requires an object of the ParametricZ Class which
        %specifies the relationship between the parameters b,c and the
        %measurements z.
        zObject;
        
    end
    
    methods
        
        % Constructor with default options
        function prob = PBiGAMPProblem()
        end
        
        %Verify that problem setup is consistent
        function check(obj)
            
            %Get dimensions from zObject
            [M,Nb,Nc] = obj.zObject.returnSizes(); %#ok<PROP>
            
            %Check them
            if (M ~= obj.M) || ...
                    (Nb ~= obj.Nb) || ...
                    (Nc ~= obj.Nc)  %#ok<PROP>
                error('PBiGAMPProblem: Sizes are not consistent')
            end
            
        end
    end
    
end
