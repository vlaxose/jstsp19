function result = checkOptions(defaultOptions,userInputs)

%checkOptions: This function returns a structure containing all the fields
%in the input userInputs. If any of the fiels in defaultOptions are not
%present in userInputs, then these fields and their values are added to
%results. if userInputs is empty, then result = defaultOptions

%Copy the user inputs
result = userInputs;

%Check for empty or non-structure
if isempty(result) || ~isstruct(userInputs)
    result = defaultOptions;
    return
end

%Otherwise, ensure that all fields are included
names = fieldnames(defaultOptions);

%Walk through fields and add any that are missing
for kk = 1:length(names)
    if ~isfield(result,names{kk})
        result.(names{kk}) = defaultOptions.(names{kk});
    end
end



    