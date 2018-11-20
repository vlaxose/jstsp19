function opt = GampRobustOpt()
% Options for the GAMP optimizer.
%
% This file is temporary to support use of GampRobustOpt until.
% The class simply instantiates GampOpt.  
opt = GampOpt();
warning('GampRobustOpt is deprecated.  Use GampOpt instead');
end

