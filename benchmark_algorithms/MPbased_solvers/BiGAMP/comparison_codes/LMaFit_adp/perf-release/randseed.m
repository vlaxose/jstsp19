function out = randseed(varargin);
%RANDSEED Generate prime numbers for use as random number seeds.
%   OUT = RANDSEED generates a random prime in the range [31, 2^17-1].
%
%   OUT = RANDSEED(STATE) generates a prime number after setting the state
%   of RAND.  This option always produces the same output for a
%   particular value of STATE.
%
%   OUT = RANDSEED(STATE,M) generates a column vector of M random primes.
%
%   OUT = RANDSEED(STATE,M,N) generates an M-by-N matrix of random primes.
%
%   OUT = RANDSEED(STATE,M,N,RMIN) generates an M-by-N matrix of random
%   primes in the range [RMIN, 2^17-1].
%
%   OUT = RANDSEED(STATE,M,N,RMIN,RMAX) generates an M-by-N matrix of random
%   primes in the range [RMIN, RMAX].
%
%   See also RAND, PRIMES.

%   Copyright 1996-2002 The MathWorks, Inc.
%   $Revision: 1.2 $  $Date: 2002/03/24 01:58:27 $

%
% Basic function setup.
%
if nargin < 1
    m = 1;
    n = 1;
    rmin = 31;
    rmax = 131071;
elseif nargin < 2
    if(~isempty(varargin{1})),
        rand('state',varargin{1});
    end
    m = 1;
    n = 1;
    rmin = 31;
    rmax = 131071;
elseif nargin < 3
    if(~isempty(varargin{1})),
        rand('state',varargin{1});
    end
    m = varargin{2};
    n = 1;
    rmin = 31;
    rmax = 131071;
elseif nargin < 4
    if(~isempty(varargin{1})),
        rand('state',varargin{1});
    end
    m = varargin{2};
    n = varargin{3};
    rmin = 31;
    rmax = 131071;
elseif nargin < 5
    if(~isempty(varargin{1})),
        rand('state',varargin{1});
    end
    m = varargin{2};
    n = varargin{3};
    rmin = varargin{4};
    rmax = 131071;
else
    if(~isempty(varargin{1})),
        rand('state',varargin{1});
    end
    m = varargin{2};
    n = varargin{3};
    rmin = varargin{4};
    rmax = varargin{5};
end
%
%  Set up a table of primes to pick from
%
primetable = primes(rmax);
primetable = primetable(primetable>=rmin);
len = length(primetable);
%
%  Expand the upper limit if there are not enough
%  primes to choose from
%
if len < m*n,
    while len < m*n,
        rmax = rmax*2;
        primetable = primes(rmax);
        primetable = primetable(primetable>=rmin);
        len = length(primetable);
    end
    warning(['Not enough primes in specified range.  Maximum value reset to ' num2str(rmax)]);
end
%
%  Use randperm to shuffle the table of primes
%
primetable = primetable(randperm(len));
%
%  Select the portion of the table required and
%  reshape the output
%
out = primetable(1:m*n);
out = reshape(out,m,n);