function [output,state]  = permute_struct( structLists , state  )
%
% cellArrayOfStructs = permute_struct( structLists);
%
% [structOut,stateOut]  = permute_struct( structLists , stateIn )
%
% The first form returns a cell array with elements that permute the 
% vectors inside each field in the structLists structure
%
% The second form permutes a structure-of-vectors, returning a new struct each 
% call until all permutations have been exhausted. The last permutation will
% be flagged by an empty stateOut.  The initial call should use the string 'init' 
%
%Example #1:
% testCases = permute_struct( struct( 'imagename',{ {'lena','cameraman' } }, 'snr' , [15 30]) );
% for k=1:length(testCases)
%   test = testCases{k};    
%   fprintf('IMAGE=%s SNR=%g\n',test.imagename,test.snr);
% end
%
%Example #2:
%   testCases = struct( 'imagename',{ {'lena','cameraman' } },'snr' , [15 30]);
%   state='init';
%   while ~isempty(state)
%       [test,state]  = permute_struct( testCases , state  );
%       fprintf('IMAGE=%s SNR=%g\n',test.imagename,test.snr);
%   end
%
%Both examples above print the following
%   IMAGE=lena SNR=15
%   IMAGE=cameraman SNR=15
%   IMAGE=lena SNR=30
%   IMAGE=cameraman SNR=30
% notice the order of permutation is controlled by the order of the fieldnames in the testCases struct
%
% Mark Borgerding (borgerding dot seven osu edu)

fn = fieldnames( structLists );
if nargin==1 
    % recursive (1-deep) call to permute the struct fields and return a single cell array with all the structs
    nPerms=0;
    state = 'init';
    while ~isempty(state)
        [~,state]  = permute_struct( structLists,state);
        nPerms=nPerms+1;
    end
    output = cell(nPerms,1);
    state = 'init';
    for k=1:nPerms
        [ output{k} ,state ] = permute_struct(structLists,state);
    end
    state='all';
    return
end

if ~isstruct(state) 
    state = struct('indices',ones(1,length(fn)) );
end

indices = state.indices;
output = struct();

desc='';
descShort='';
for k=1:length(fn)
    vec = getfield(structLists,fn{k});
    if iscell(vec)
        val = vec{indices(k)};
    else
        val = vec(indices(k));
    end

    fieldDesc = [fn{k} '=' num2str(val) ' '];
    desc = [ desc fieldDesc ];
    if numel(vec) > 1
        descShort = [ descShort fieldDesc ];  % short description only names the parts that vary from one trial to the next
    end
    output = setfield(output,fn{k},val);
end
output.description = desc;
output.short_description = descShort;

lastPerm=true;
for k=1:length(indices)
    vec = getfield(structLists,fn{k});
    indices(k) = indices(k) + 1;
    if indices(k) <= length(vec)
        lastPerm=false;
        break;
    end
    indices(k) = 1; %and increment the next field
end
if lastPerm
    state = [];
else
    state = setfield(state,'indices',indices);
end

