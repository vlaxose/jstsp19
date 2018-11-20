% DemoGUI
%
% This function will open a GUI that can be used to configure and execute a
% GAMP solver that infers a length-N random vector x from a length-M
% observation vector y, genertaed via the Markov chain
%                            x -> z = A*x -> y,
% where A is a linear operator, and p(x) and p(y|z) are scalar-separable
% distributions (i.e., they factor into a product of scalar pdfs).
%

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: zinielj@ece.osu.edu
% Last change: 11/26/12
% Change summary: 
%       - Created (11/24/12; JAZ)
% Version 0.1
%

function DemoGUI()

global gui data;

% Make sure that the user has included all necessary directories in
% MATLAB's search path
if ~exist('gampEst.m', 'file')
    % "main" subdirectory missing
    msgbox(['Please add the "main" gampmatlab subdirectory to MATLAB''s' ...
        ' search path'], 'Directory missing', 'error')
    return
elseif ~exist('isHGUsingMATLABClasses.p', 'file')
    % GUI->Patch subdirectory missing
    GUIname = mfilename;
    GUIdir = mfilename('fullpath');
    GUIdir = GUIdir(1:end-numel(GUIname));
    addpath([GUIdir filesep 'Patch']);     % Add Patch subdirectory to path
end

% Data is shared between all child functions by declaring the variables
% here (they become global to the function). We keep things tidy by putting
% all GUI stuff in one structure and all data stuff in a GUIData object
gui = struct();             % Empty structure for GUI handles
data = GUIData();          % Create a default object of the GUIData class

% Initialize the GUI interface
createInterface();

% Explicitly call the demo display so that it gets included if we deploy
displayEndOfDemoMessage('')




%*************************************************************************%
%                       CREATE INTERFACE FUNCTION                         %
%*************************************************************************%
% This function initializes the GUI interface at startup, placing
% components where they belong
function createInterface()
    % Create the user interface for the application and return a
    % structure of handles for global use.
%     global gui;
    
    % Setup screen sizing
    DesiredSize = [900, 530];   % Desired GUI dimensions (W x H, px)
    ScreenSize = get(0, 'screensize');      % Current screen resolution
    ScreenSize = ScreenSize(3:4);
    GUISize = min(DesiredSize, ScreenSize); % Allowable GUI size
    Offset = [0, 0];                        % Offset from lower-left corner
    if ScreenSize(1) - GUISize(1) > 100
        Offset(1) = 100;
    end
    if ScreenSize(2) - GUISize(2) > 100
        Offset(2) = 100;
    end

    % Open a window and add some menus
    gui.Window = figure( ...
        'Name', 'GAMP GUI', ...
        'NumberTitle', 'off', ...
        'MenuBar', 'none', ...
        'Toolbar', 'figure', ...
        'Position', [Offset, GUISize], ...
        'HandleVisibility', 'off' );

    % Set default panel color
    uiextras.set( gui.Window, 'DefaultBoxPanelTitleColor', [0.7 1.0 0.7] );

    % + File menu
    gui.FileMenu = uimenu( gui.Window, 'Label', 'File' );
    uimenu( gui.FileMenu, 'Label', 'Reset', 'Callback', @onReset);
    uimenu( gui.FileMenu, 'Label', 'Exit', 'Callback', @onExit );

%     % + View menu
%     gui.ViewMenu = uimenu( gui.Window, 'Label', 'View' );
%     for ii=1:numel( data )
%         uimenu( gui.ViewMenu, 'Label', data{ii}, 'Callback', @onMenuSelection );
%     end

    % + Help menu
    helpMenu = uimenu( gui.Window, 'Label', 'Help' );
    uimenu( helpMenu, 'Label', 'Documentation', 'Callback', @onHelp );
    
    % + Create tabbed panel
    gui.TabPanel = uiextras.TabPanel('Parent', gui.Window, 'Padding', 5, ...
        'TabSize', 100, 'Callback', @onTabSelection);
    
    % Create and populate the first/main/execute tab
    createMainTab();
    
    % Create placeholders for remaining tabs (but do not populate tabs at
    % startup in order to create more responsive initial load)
    gui.DatasetPanel = uiextras.Panel( 'Parent', gui.TabPanel, ...
        'Title', 'Dataset Specification', 'Tag', 'Dataset:Panel');
    gui.EstimInPanel = uiextras.Panel( 'Parent', gui.TabPanel, ...
        'Title', 'EstimIn Configuration', 'Tag', 'EstimIn:Panel');
    gui.EstimOutPanel = uiextras.Panel( 'Parent', gui.TabPanel, ...
        'Title', 'EstimOut Configuration', 'Tag', 'EstimOut:Panel');
    gui.OptionsPanel = uiextras.Panel( 'Parent', gui.TabPanel, ...
        'Title', 'GAMP Options', 'Tag', 'RunOptions:Panel');
    
    % Name and select tabs
    gui.TabPanel.TabNames = {'Execute', 'Dataset', 'EstimIn', ...
        'EstimOut', 'Run Options'};
    gui.TabPanel.SelectedChild = 1;     % Start on main/execution tab
    
    % Use timer to populate remaining tabs without causing lag when GUI
    % first becomes visible
    startuptimer = timer('ExecutionMode', 'SingleShot', 'TimerFcn', ...
        @populateTabs, 'StartDelay', .5);
    start(startuptimer);
    while strcmp(get(startuptimer, 'Running'), 'on')
        pause(.1);
    end
    delete(startuptimer);

end % createInterface


%*************************************************************************%
%                         POPULATE TABS FUNCTION                          %
%*************************************************************************%
function populateTabs(~, ~)
    createDatasetTab();         % Dataset specification tab
    createEstimInTab(0);      	% EstimIn model tab
    createEstimOutTab();        % EstimOut channel tab
    createOptionsTab();         % Runtime options tab
end


%*************************************************************************%
%                       CREATE MAIN TAB FUNCTION                          %
%*************************************************************************%
% This function is used to populate the first tab (the main/run tab) of the
% GUI with the various containers and UI controls in it
function createMainTab()
    
    % Arrange the main interface
    gui.mainLayout = uiextras.GridFlex( 'Parent', gui.TabPanel, 'Spacing', 3 );

    % + Create the panels that populate the main interface
    configPanel = uiextras.Panel( 'Parent', gui.mainLayout, 'Title', ...
        'GAMP Configuration' );    % Specifies TurboOpt choices
    controlPanel = uiextras.Panel( 'Parent', gui.mainLayout, 'Title', ...
        'Execution Controls' );         % Buttons for running classifier(s)
    gui.PlotPanel = uiextras.Panel( 'Parent', gui.mainLayout, 'Title', ...
        'Recovery Plot' );              % Panel for plotting weight vectors
    statPanel = uiextras.Panel( 'Parent', gui.mainLayout, 'Title', ...
        'Recovery Statistics' );        % Performance statistics

    % + Adjust the main layout orientation
    set( gui.mainLayout, 'ColumnSizes', [-1, -3], 'RowSizes', [-2, -1] );
    
    % + Create the turboGAMP configuration components
    configLayout = uiextras.VBox( 'Parent', configPanel, 'Padding', 3, ...
        'Spacing', 3);
    gui.EstimInPopup = uicontrol('Style', 'PopUpMenu', 'Parent', ...
        configLayout, 'String', ...
        ['[EstimIn Class]'; data.EstimInUIData.Names(:)], 'Value', 1, ...
        'ToolTipString', 'Select the GAMP EstimIn class (signal prior)', ...
        'Callback', @onEstimInPopupSelection);
    gui.SparsifyCheckbox = uicontrol('Style', 'Checkbox', 'Parent', ...
        configLayout, 'String', 'Sparsify chosen EstimIn class?', ...
        'TooltipString', sprintf('%s \n', 'Checking this box will cause the', ...
        'chosen EstimIn class above to be', 'wrapped in a SparseScaEstim wrapper', ...
        'which encourages sparse solutions via', 'a Bernoulli-(EstimIn) prior, e.g.,', ...
        'a Gaussian prior EstimIn class becomes', 'a Bernoulli-Gaussian prior'), ...
        'Callback', @onSparsifyCheckboxSelection, 'Value', 0);
    gui.SparsityBar = uiextras.HBox('Parent', configLayout);
    uicontrol('Style', 'Text', 'Parent', gui.SparsityBar, 'String', ...
        'Sparsity rate [0,1]:');
    gui.SparsityRateEditBox = uicontrol('Style', 'Edit', 'Parent', ...
        gui.SparsityBar, 'String', num2str(data.EstimInUIData.SparsityRate), ...
        'TooltipString', ...
        'Enter the (scalar) prior probability that coefficient is non-zero', ...
        'Callback', @onSparsityRateEditActivity);
    set(gui.SparsityBar, 'Sizes', [-2, -1])
    set(gui.SparsityBar, 'Visible', 'off');
    gui.EstimOutPopup = uicontrol('Style', 'PopUpMenu', 'Parent', ...
        configLayout, 'String', ...
        ['[EstimOut Class]'; data.EstimOutUIData.Names(:)], 'Value', 1, ...
        'ToolTipString', 'Select the GAMP EstimOut class (observation model)', ...
        'Callback', @onEstimOutPopupSelection);
    uiextras.Empty('Parent', configLayout);
    set(configLayout, 'Sizes', [25, 25, 25, 25, -5])
    
    % + Create the execution controls
    controlLayout = uiextras.VBox( 'Parent', controlPanel, ...
        'Padding', 3, 'Spacing', 3 );
    gui.RunButton = uicontrol( 'Style', 'PushButton', ...
        'Parent', controlLayout, 'String', 'Run GAMP', ...
        'ToolTipString', ...
        'Press to run GAMP and display results', ...
        'Callback', @onRunGAMP );
    gui.ClearButton = uicontrol( 'Style', 'PushButton', ...
        'Parent', controlLayout, 'String', 'Clear', ...
        'ToolTipString', 'Press to clear statistics and plot', ...
        'Callback', @onClear );
    gui.ResetButton = uicontrol( 'Style', 'PushButton', ...
        'Parent', controlLayout, 'String', 'Reset', ...
        'ToolTipString', 'Press to reset all settings to defaults', ...
        'Callback', @onReset );
    gui.saveData = uicontrol('Style', 'Checkbox', 'Parent', controlLayout, ...
        'String', 'Save variables to workspace?', 'TooltipString', ...
        sprintf('%s \n', 'Enable this checkbox if you would like the GUI to save ', ...
        'important variables, such as the observations ', ...
        '(''y''), the measurement matrix (as a LinTrans object) ', ...
        '(''A''), the GAMP recovered signal (''x_gamp''), and ', ...
        '(possibly) the synthetic "ground truth" signal, ', ...
        '(''x_true''), to the workspace'));
%     set( controlLayout, 'Sizes', [-1 28] ); % Make the list fill the space

    % + Create the recovery plot panel
    gui.ViewAxes = axes( 'Parent', gui.PlotPanel );
    
    % + Create the recovery statistics table
    RowNames = {'Normalized MSE (dB)', 'MSE', 'Signal Dimension (N)', ...
        'Observation Dimension (M)', 'Support Cardinality (K)', ...
        'Runtime (s)'};
    ColNames = {'Truth', 'GAMP'};
    gui.StatTable = uitable( 'Parent', statPanel, 'ColumnName', ...
        ColNames, 'RowName', RowNames, 'RearrangeableColumns', 'on');
end


%*************************************************************************%
%                       CREATE DATASET TAB FUNCTION                       %
%*************************************************************************%
% This function is used to populate the dataset tab of the GUI with the 
% various containers and UI controls in it
function createDatasetTab()
%     gui.DatasetPanel = uiextras.Panel( 'Parent', gui.TabPanel, ...
%         'Title', 'Dataset Specification', 'Tag', 'Dataset:Panel');
    
    % Create outer vertical box to segregate synthetic/workspace dataset
    % radio buttons from rest of panel
    OuterVBox = uiextras.VBox('Parent', gui.DatasetPanel);
    
    % Create a horizontal button box and place in it two radio buttons, one
    % to enable the synthetic dataset parameters, the other to enable the
    % workspace dataset parameters
%     DatasetHiHBox = uiextras.HButtonBox('Parent', OuterVBox, 'Padding', 5, ...
%         'ButtonSize', [150, 20]);
    DatasetHiHBox = uiextras.HBox('Parent', OuterVBox, 'Padding', 5);
    gui.SynthRadioBtn = uicontrol('Parent', DatasetHiHBox, 'Style', ...
        'RadioButton', 'String', 'Synthetic Dataset', 'Value', 1, ...
        'ToolTipString', 'Press to generate synthetic data', ...
        'Callback', @onSynthDataSelection);
    gui.WSpcRadioBtn = uicontrol('Parent', DatasetHiHBox, 'Style', ...
        'RadioButton', 'String', 'Workspace Dataset', 'Value', 0, ...
        'ToolTipString', 'Press to select data from the workspace', ...
        'Callback', @onWSpcDataSelection);
    
    % Create a horizontal box layout of two panels.  The first panel will
    % contain synthetic dataset parameters, the second will contain
    % workspace dataset parameters
    DatasetLowHBox = uiextras.HBox('Parent', OuterVBox, 'Padding', 5);
    gui.SynthPanel = uiextras.Panel('Parent', DatasetLowHBox, 'Padding', 8, ...
        'Title', 'Synthetic Dataset Setup', 'Tag', 'Dataset:Panel:Synthetic');
    gui.WSpcPanel = uiextras.Panel('Parent', DatasetLowHBox, 'Padding', 8, ...
        'Title', 'Workspace Dataset Setup', 'Tag', 'Dataset:Panel:Workspace', ...
        'Enable', 'off');
    
    % Set size of outer vertical box containers
    set(OuterVBox, 'Sizes', [30, -5]);

    % In the (leftmost) panel of dataset options, create a grid that will
    % contain labels and UI controls
    SynthVBox = uiextras.VBox('Parent', gui.SynthPanel);
    gui.SynthDataGrid = uiextras.Grid('Parent', SynthVBox);

    % Add UI components to this panel
    Ncomp = size(data.DatasetUIData.SynthParams, 1);  % # of UI components to add
    for j = 1:Ncomp
        % In first column of grid, place labels for each of the
        % parameters
        tmp = uicontrol('Parent', gui.SynthDataGrid, 'String', ...
            data.DatasetUIData.SynthParams{j,3}, 'Style', 'Text');
        if ischar(data.DatasetUIData.SynthParams{j,4})
            set(tmp, 'TooltipString', data.DatasetUIData.SynthParams{j,4});
        end
    end

    for j = 1:Ncomp
        % In second column of grid, identify the UI control to add, and
        % place it
        switch lower(data.DatasetUIData.SynthParams{j,2})
            case 'edit'
                % Add an edit box for user to insert numeric data, and
                % initialize with the default value for that parameter
                uicontrol('Parent', gui.SynthDataGrid, 'Style', ...
                    'Edit', 'String', data.DatasetUIData.SynthValues{j,2}, ...
                    'Tag', data.DatasetUIData.SynthParams{j,1}, ...
                    'TooltipString', data.DatasetUIData.SynthParams{j,4});
            case 'checkbox'
                % Add a checkbox, and initialize with default value
                tmp = uicontrol('Parent', gui.SynthDataGrid, 'Style', ...
                    'Checkbox', 'Value', data.DatasetUIData.SynthValues{j,2}, ...
                    'Tag', data.DatasetUIData.SynthParams{j,1}, ...
                    'TooltipString', data.DatasetUIData.SynthParams{j,4});
                % Special handling for one checkbox
                if strcmp(get(tmp, 'Tag'), 'GenDiff')
                    set(tmp, 'Callback', @onNonEstimInGen);
                end
            case 'popupmenu'
                % Add a pop-up menu
                uicontrol('Parent', gui.SynthDataGrid, 'Style', ...
                    'popupmenu', 'String', data.DatasetUIData.SynthParams{j,4}, ...
                    'Value', data.DatasetUIData.SynthValues{j,2}, ...
                    'Tag', data.DatasetUIData.SynthParams{j,1});
            otherwise
                error('Unrecognized UI control ''%s'' for Dataset tab', ...
                    data.DatasetUIData.SynthParams{j,2})
         end
    end
    
    % Add third column of empty filler to make things look
    % presentable
    for j = 1:Ncomp
        uiextras.Empty('Parent', gui.SynthDataGrid);
    end
    
    % Finally, reshape the grid so that it knows how to arrage sizes
    set(gui.SynthDataGrid, 'ColumnSizes', [200, 150, -1], 'RowSizes', ...
        20*ones(1,Ncomp))
    
    % Beneath the basic parameters, create a sub-panel just for the
    % (optional) non-EstimIn class choice of generating distribution for x
    gui.AltEstimInPanel = uiextras.Panel('Parent', SynthVBox, 'Padding', 5, ...
        'Title', 'Alternative Generating Distribution', 'Tag', ...
        'Dataset:Panel:Synthetic:AltPanel', 'Enable', 'off');
    AltVBox = uiextras.VBox('Parent', gui.AltEstimInPanel);
    gui.AltEstimInPopup = uicontrol('Style', 'PopUpMenu', 'Parent', ...
        AltVBox, 'String', ...
        ['[Alternative EstimIn Class]'; data.EstimInUIData.Names(:)], 'Value', 1, ...
        'ToolTipString', 'Select the generating distribution EstimIn class', ...
        'Callback', @onAltEstimInPopupSelection);
    gui.AltSparsifyCheckbox = uicontrol('Style', 'Checkbox', 'Parent', ...
        AltVBox, 'String', 'Sparsify chosen EstimIn class?', ...
        'TooltipString', sprintf('%s \n', 'Checking this box will cause the', ...
        'chosen EstimIn class above to be', 'wrapped in a SparseScaEstim wrapper', ...
        'which encourages sparse solutions via', 'a Bernoulli-(EstimIn) prior, e.g.,', ...
        'a Gaussian prior EstimIn class becomes', 'a Bernoulli-Gaussian prior'), ...
        'Callback', @onAltSparsifyCheckboxSelection, 'Value', 0);
    gui.AltSparsityBar = uiextras.HBox('Parent', AltVBox);
    uicontrol('Style', 'Text', 'Parent', gui.AltSparsityBar, 'String', ...
        'Sparsity rate [0,1]:');
    gui.AltSparsityRateEditBox = uicontrol('Style', 'Edit', 'Parent', ...
        gui.AltSparsityBar, 'String', num2str(data.EstimInUIData.SparsityRate), ...
        'TooltipString', ...
        'Enter the (scalar) prior probability that coefficient is non-zero', ...
        'Callback', @onAltSparsityRateEditActivity);
    set(gui.AltSparsityBar, 'Visible', 'off');
    gui.AltSynthPanel = uiextras.Panel('Parent', AltVBox);
    createEstimInTab(1);    % Create sub-panel for alternative EstimIn params
    set(AltVBox, 'Sizes', [25, 25, 25, -1]);
    set(gui.AltEstimInPanel, 'Enable', 'off');
    set(SynthVBox, 'Sizes', [-1, -2])
    
    % In the (rightmost) panel of GAMP options, create a grid that will
    % contain labels and UI controls
    WSpcVBox = uiextras.VBox('Parent', gui.WSpcPanel);
    gui.WSpcDataGrid = uiextras.Grid('Parent', WSpcVBox);
    gui.RefreshListBtn = uicontrol('Parent', WSpcVBox, 'Style', ...
        'PushButton', 'String', 'Refresh Lists', 'TooltipString', ...
        'Press to refresh the lists with the current workspace variables', ...
        'Callback', @onRefreshListSelection);
    set(WSpcVBox, 'Sizes', [-1, 25]);   % Adjust sizing
    
    % Create a 2-by-1 array of listboxes, each containing the names of all
    % current variables in the base workspace
    
    uicontrol('Parent', gui.WSpcDataGrid, 'Style', 'Text', 'String', ...
        'Select the vector of observations, y', 'TooltipString', ...
        'Select the workspace variable containing the observations')
    gui.yListBox = uicontrol('Parent', gui.WSpcDataGrid, 'Style', ...
        'ListBox', 'Max', 1, 'Min', 1, 'TooltipString', ...
        'Select the workspace variable containing the observations', ...
        'Callback', @onListBoxSelection, 'Tag', 'y');
    uiextras.Empty('Parent', gui.WSpcDataGrid);
    uicontrol('Parent', gui.WSpcDataGrid, 'Style', 'Text', 'String', ...
        'Select measurement matrix, A', 'TooltipString', ...
        sprintf('%s\n %s\n', 'Select the workspace variable containing the ', ...
        'measurement matrix (or valid LinTrans object)'))
    gui.AListBox = uicontrol('Parent', gui.WSpcDataGrid, 'Style', ...
        'ListBox', 'Max', 1, 'Min', 1, 'TooltipString', ...
        sprintf('%s\n %s\n', 'Select the workspace variable containing the ', ...
        'measurement matrix (or valid LinTrans object)'), ...
        'Callback', @onListBoxSelection, 'Tag', 'A');
    uiextras.Empty('Parent', gui.WSpcDataGrid);
    uicontrol('Parent', gui.WSpcDataGrid, 'Style', 'Text', 'String', ...
        '[Optional] Select ground truth, x_true', 'TooltipString', ...
        sprintf('%s\n %s\n', '(Optionally) select the workspace variable ', ...
        'containing the true unknown signal'))
    gui.xtrueListBox = uicontrol('Parent', gui.WSpcDataGrid, 'Style', ...
        'ListBox', 'Max', 1, 'Min', 1, 'TooltipString', ...
        sprintf('%s\n %s\n', '(Optionally) select the workspace variable ', ...
        'containing the true unknown signal'), ...
        'Callback', @onListBoxSelection, 'Tag', 'x_true');
    
    % Finally, reshape the grid so that it knows how to arrage sizes
    set(gui.WSpcDataGrid, 'ColumnSizes', -1, 'RowSizes', ...
        [20, -2, 15, 20, -2, 15, 20, -2])
    
    % Disable workspace panel at initialization
    set(gui.WSpcPanel, 'Enable', 'off');
end


%*************************************************************************%
%                       CREATE ESTIMIN TAB FUNCTION                       %
%*************************************************************************%
% This function is used to populate the EstimIn tab of the GUI with the 
% various containers and UI controls in it
function createEstimInTab(val)
    % Create the panel container for this tab
    if val == 0
        % Create panel for EstimIn tab
        BigParent = gui.EstimInPanel;
        CardPanelName = 'EstimInCardPanel';
        CardNames = 'EstimInCards';
    elseif val == 1
        % Create panel for Dataset alternative generating EstimIn
        BigParent = gui.AltSynthPanel;
        CardPanelName = 'AltEstimInCardPanel';
        CardNames = 'AltEstimInCards';
    end
    
    % Create the card panel for this tab (i.e., a card for each possible
    % EstimIn class contained in the GUIData object, data, with only one
    % card visible on screen at a time)
    Ncards = numel(data.EstimInUIData.Names);    % # of cards
    gui.(CardPanelName) = uiextras.CardPanel( 'Parent', BigParent, ...
        'Padding', 5 );
    gui.(CardNames) = cell(1,Ncards);
    % Iterate through each card, placing a single VBox UI container in each
    % card that will hold all of the UI components for that card, i.e., all
    % of the relevant interfaces to specify the values of the parameters
    % that make up a particular EstimIn class object
    for i = 1:Ncards
        % Create a grid for the parameter labels and their associated UI
        % components
        gui.(CardNames){i} = uiextras.Grid('Parent', gui.(CardPanelName));
        
        % Add a label to this card identifying the particular EstimIn class
        % that is visible
        uicontrol('Parent', gui.(CardNames){i}, 'Style', 'Text', 'String', ...
            'EstimIn Class: ', 'FontSize', 12, 'HorizontalAlignment', 'Right');
        % Add UI components to this EstimIn card Grid
        Ncomp = size(data.EstimInUIData.Params{i}, 1);  % # of UI components to add
        for j = 1:Ncomp
            % In first column of grid, place labels for each of the
            % parameters
            tmp = uicontrol('Parent', gui.(CardNames){i}, 'String', ...
                data.EstimInUIData.Params{i}{j,3}, 'Style', 'Text', ...
                'HorizontalAlignment', 'Right');
            if ischar(data.EstimInUIData.Params{i}{j,4})
                set(tmp, 'TooltipString', data.EstimInUIData.Params{i}{j,4});
            end
        end
        % Add empty filler column at bottom of column
        uiextras.Empty('Parent', gui.(CardNames){i});
        
        % Add a label to this card identifying the particular EstimIn class
        % that is visible
        uicontrol('Parent', gui.(CardNames){i}, 'Style', 'Text', 'String', ...
            data.EstimInUIData.Names{i}, 'FontSize', 12);
        for j = 1:Ncomp
            % In second column of grid, identify the UI control to add, and
            % place it
            switch lower(data.EstimInUIData.Params{i}{j,2})
                case 'edit'
                    % Add an edit box for user to insert numeric data, and
                    % initialize with the default value for that parameter
                    uicontrol('Parent', gui.(CardNames){i}, 'Style', ...
                        'Edit', 'String', data.EstimInUIData.Values{i}{j,2}, ...
                        'Tag', data.EstimInUIData.Params{i}{j,1}, ...
                        'TooltipString', data.EstimInUIData.Params{i}{j,4});
                case 'checkbox'
                    % Add a checkbox, and initialize with default value
                    uicontrol('Parent', gui.(CardNames){i}, 'Style', ...
                        'Checkbox', 'Value', data.EstimInUIData.Values{i}{j,2}, ...
                        'Tag', data.EstimInUIData.Params{i}{j,1}, ...
                        'TooltipString', data.EstimInUIData.Params{i}{j,4});
                case 'popupmenu'
                    % Add a pop-up menu
                    uicontrol('Parent', gui.(CardNames){i}, 'Style', ...
                        'popupmenu', 'String', data.EstimInUIData.Params{i}{j,4}, ...
                        'Value', data.EstimInUIData.Values{i}{j,2}, ...
                        'Tag', data.EstimInUIData.Params{i}{j,1});
                otherwise
                    error('Unrecognized UI control %s for %s EstimIn class', ...
                        data.EstimInUIData.Params{i}{j,2}, ...
                        data.EstimInUIData.Names{i})
            end
        end
        % Add empty filler column at bottom of column
        uiextras.Empty('Parent', gui.(CardNames){i});
        
        % Add third column of empty filler to make things look
        % presentable
        for j = 1:Ncomp+2
            uiextras.Empty('Parent', gui.(CardNames){i});
        end
            
        % Finally, reshape the grid so that it knows how to arrage sizes
        if val
            set(gui.(CardNames){i}, 'ColumnSizes', [225, 170, -4], 'RowSizes', ...
                [20*ones(1,Ncomp+1), -9])
        else
            set(gui.(CardNames){i}, 'ColumnSizes', [-2.5, -2, -7], 'RowSizes', ...
            [20*ones(1,Ncomp+1), -9])
        end
    end
    gui.(CardPanelName).SelectedChild = 1;
end


%*************************************************************************%
%                       CREATE ESTIMOUT TAB FUNCTION                      %
%*************************************************************************%
% This function is used to populate the EstimOut tab of the GUI with the 
% various containers and UI controls in it
function createEstimOutTab()
    % Create the panel container for this tab
%     gui.EstimOutPanel = uiextras.Panel( 'Parent', gui.TabPanel, ...
%         'Title', 'EstimOut Model', 'Tag', 'EstimOut:Panel');
    
    % Create the card panel for this tab (i.e., a card for each possible
    % EstimOut class contained in the GUIData object, data, with only one
    % card visible on screen at a time)
    Ncards = numel(data.EstimOutUIData.Names);    % # of cards
    gui.EstimOutCardPanel = uiextras.CardPanel( 'Parent', ...
        gui.EstimOutPanel, 'Padding', 5 );
    gui.EstimOutCards = cell(1,Ncards);
    % Iterate through each card, placing a single VBox UI container in each
    % card that will hold all of the UI components for that card, i.e., all
    % of the relevant interfaces to specify the values of the parameters
    % that make up a particular EstimOut class object
    for i = 1:Ncards
        % Create a grid for the parameter labels and their associated UI
        % components
        gui.EstimOutCards{i} = uiextras.Grid('Parent', ...
            gui.EstimOutCardPanel);
        
        % Add a label to this card identifying the particular EstimOut 
        % class that is visible
        uicontrol('Parent', gui.EstimOutCards{i}, 'Style', 'Text', ...
            'String', 'EstimOut Class: ', 'FontSize', 12, ...
            'HorizontalAlignment', 'Right');
        % Add UI components to this EstimOut card Grid
        Ncomp = size(data.EstimOutUIData.Params{i}, 1);  % # of UI components to add
        for j = 1:Ncomp
            % In first column of grid, place labels for each of the
            % parameters
            if ~strcmpi(data.EstimOutUIData.Params{i}{j,2}, 'meas')
                % Not associated with measurement storage field
                tmp = uicontrol('Parent', gui.EstimOutCards{i}, 'String', ...
                    data.EstimOutUIData.Params{i}{j,3}, 'Style', 'Text', ...
                    'HorizontalAlignment', 'Right');
                if ischar(data.EstimOutUIData.Params{i}{j,4})
                    set(tmp, 'TooltipString', data.EstimOutUIData.Params{i}{j,4});
                end
            end
        end
        % Add empty filler column at bottom of column
        uiextras.Empty('Parent', gui.EstimOutCards{i});
        
        % Add a label to this card identifying the particular EstimOut class
        % that is visible
        uicontrol('Parent', gui.EstimOutCards{i}, 'Style', 'Text', 'String', ...
            data.EstimOutUIData.Names{i}, 'FontSize', 12);
        for j = 1:Ncomp
            % In second column of grid, identify the UI control to add, and
            % place it
            switch lower(data.EstimOutUIData.Params{i}{j,2})
                case 'edit'
                    % Add an edit box for user to insert numeric data, and
                    % initialize with the default value for that parameter
                    uicontrol('Parent', gui.EstimOutCards{i}, 'Style', ...
                        'Edit', 'String', data.EstimOutUIData.Values{i}{j,2}, ...
                        'Tag', data.EstimOutUIData.Params{i}{j,1}, ...
                        'TooltipString', data.EstimOutUIData.Params{i}{j,4});
                case 'checkbox'
                    % Add a checkbox, and initialize with default value
                    uicontrol('Parent', gui.EstimOutCards{i}, 'Style', ...
                        'Checkbox', 'Value', data.EstimOutUIData.Values{i}{j,2}, ...
                        'Tag', data.EstimOutUIData.Params{i}{j,1}, ...
                        'TooltipString', data.EstimOutUIData.Params{i}{j,4});
                case 'popupmenu'
                    % Add a pop-up menu
                    uicontrol('Parent', gui.EstimOutCards{i}, 'Style', ...
                        'popupmenu', 'String', data.EstimOutUIData.Params{i}{j,4}, ...
                        'Value', data.EstimOutUIData.Values{i}{j,2}, ...
                        'Tag', data.EstimOutUIData.Params{i}{j,1});
                case 'meas'
                    % This is the field that will hold actual measurements,
                    % so don't place anything here
                otherwise
                    error('Unrecognized UI control %s for %s EstimOut class', ...
                        data.EstimOutUIData.Params{i}{j,2}, ...
                        data.EstimOutUIData.Names{i})
            end
        end
        % Add empty filler column at bottom of column
        uiextras.Empty('Parent', gui.EstimOutCards{i});
        
        % Add third column of empty filler to make things look
        % presentable
        uiextras.Empty('Parent', gui.EstimOutCards{i});
        for j = 1:Ncomp
            if ~strcmpi(data.EstimOutUIData.Params{i}{j,2}, 'meas')
                % Not associated with measurement storage field
                uiextras.Empty('Parent', gui.EstimOutCards{i});
            else
                Ncomp = Ncomp - 1;
            end
        end
        uiextras.Empty('Parent', gui.EstimOutCards{i});
            
        % Finally, reshape the grid so that it knows how to arrage sizes
        set(gui.EstimOutCards{i}, 'ColumnSizes', [-2.5, -2, -7], 'RowSizes', ...
            [20*ones(1,Ncomp+1), -9])
    end
    gui.EstimOutCardPanel.SelectedChild = 1;
end



%*************************************************************************%
%                       CREATE OPTIONS TAB FUNCTION                       %
%*************************************************************************%
% This function is used to populate the dataset tab of the GUI with the 
% various containers and UI controls in it
function createOptionsTab()
%     gui.OptionsPanel = uiextras.Panel( 'Parent', gui.TabPanel, ...
%         'Title', 'Runtime Options', 'Tag', 'RunOptions:Panel');
    
    % Create a vertical box layout of three panels.  The first panel will
    % contain basic GAMP parameters, the second will contain intermediate 
    % GAMP parameters
    OptionsHBox = uiextras.VBox('Parent', gui.OptionsPanel, 'Padding', 5);
    BasicGAMPPanel = uiextras.Panel('Parent', OptionsHBox, 'Padding', 5, ...
        'Title', 'Basic GAMP Options', 'Tag', 'RunOptions:Panel:Basic');
    ImdGAMPPanel = uiextras.Panel('Parent', OptionsHBox, 'Padding', 5, ...
        'Title', 'Intermediate GAMP Options', 'Tag', 'RunOptions:Panel:Intermediate');
    AdvGAMPPanel = uiextras.Panel('Parent', OptionsHBox, 'Padding', 5, ...
        'Title', 'Advanced GAMP Options', 'Tag', 'RunOptions:Panel:Advanced');
    set(OptionsHBox, 'Sizes', [-1, -1, -1])

    % In the (topmost) panel of GAMP options, create a grid that will
    % contain labels and UI controls
    gui.GAMPOptionsGrid{1} = uiextras.Grid('Parent', BasicGAMPPanel);
    gui.GAMPOptionsGrid{2} = uiextras.Grid('Parent', ImdGAMPPanel);
    gui.GAMPOptionsGrid{3} = uiextras.Grid('Parent', AdvGAMPPanel);

    % Add UI components to this RunOptions panel
    Ncomp = size(data.GAMPOptUIData.GAMPParams, 1);  % # of UI components to add
    for j = 1:Ncomp
        % In first column of grid, place labels for each of the
        % parameters
        loc = data.GAMPOptUIData.GAMPParams{j,5};
        tmp = uicontrol('Parent', gui.GAMPOptionsGrid{loc}, 'String', ...
            data.GAMPOptUIData.GAMPParams{j,3}, 'Style', 'Text', ...
            'HorizontalAlignment', 'Right');
        if ischar(data.GAMPOptUIData.GAMPParams{j,4})
            set(tmp, 'TooltipString', data.GAMPOptUIData.GAMPParams{j,4});
        end
    end
    % Add empty filler column at bottom of each column
    uiextras.Empty('Parent', gui.GAMPOptionsGrid{1});
    uiextras.Empty('Parent', gui.GAMPOptionsGrid{2});
    uiextras.Empty('Parent', gui.GAMPOptionsGrid{3});

    for j = 1:Ncomp
        loc = data.GAMPOptUIData.GAMPParams{j,5};
        
        % In second column of grid, identify the UI control to add, and
        % place it
        switch lower(data.GAMPOptUIData.GAMPParams{j,2})
            case 'edit'
                % Add an edit box for user to insert numeric data, and
                % initialize with the default value for that parameter
                uicontrol('Parent', gui.GAMPOptionsGrid{loc}, 'Style', ...
                    'Edit', 'String', data.GAMPOptUIData.GAMPValues{j,2}, ...
                    'Tag', data.GAMPOptUIData.GAMPParams{j,1}, ...
                    'TooltipString', data.GAMPOptUIData.GAMPParams{j,4});
            case 'checkbox'
                % Add a checkbox, and initialize with default value
                uicontrol('Parent', gui.GAMPOptionsGrid{loc}, 'Style', ...
                    'Checkbox', 'Value', data.GAMPOptUIData.GAMPValues{j,2}, ...
                    'Tag', data.GAMPOptUIData.GAMPParams{j,1}, ...
                    'TooltipString', data.GAMPOptUIData.GAMPParams{j,4});
            case 'popupmenu'
                % Add a pop-up menu
                uicontrol('Parent', gui.turboOptionsGrid{loc}, 'Style', ...
                    'popupmenu', 'String', data.GAMPOptUIData.GAMPParams{j,4}, ...
                    'Value', data.GAMPOptUIData.GAMPValues{j,2}, ...
                    'Tag', data.GAMPOptUIData.GAMPParams{j,1});
            otherwise
                error('Unrecognized UI control %s for Options tab', ...
                    data.GAMPOptUIData.GAMPParams{j,2})
        end
    end
    % Add empty filler column at bottom of column
    uiextras.Empty('Parent', gui.GAMPOptionsGrid{1});
    uiextras.Empty('Parent', gui.GAMPOptionsGrid{2});
    uiextras.Empty('Parent', gui.GAMPOptionsGrid{3});
    
    % Add third column of empty filler to make things look
    % presentable
    for j = 1:Ncomp
        loc = data.GAMPOptUIData.GAMPParams{j,5};
        uiextras.Empty('Parent', gui.GAMPOptionsGrid{loc});
    end
    uiextras.Empty('Parent', gui.GAMPOptionsGrid{1});
    uiextras.Empty('Parent', gui.GAMPOptionsGrid{2});
    uiextras.Empty('Parent', gui.GAMPOptionsGrid{3});
    
    % Finally, reshape the grid so that it knows how to arrage sizes
    set(gui.GAMPOptionsGrid{1}, 'ColumnSizes', [225, 80, -2], 'RowSizes', ...
        [20*ones(1,sum([data.GAMPOptUIData.GAMPParams{:,5}] == 1)), -5])
    set(gui.GAMPOptionsGrid{2}, 'ColumnSizes', [225, 80, -2], 'RowSizes', ...
        [20*ones(1,sum([data.GAMPOptUIData.GAMPParams{:,5}] == 2)), -5])
    set(gui.GAMPOptionsGrid{3}, 'ColumnSizes', [225, 80, -2], 'RowSizes', ...
        [20*ones(1,sum([data.GAMPOptUIData.GAMPParams{:,5}] == 3)), -5])
end


%*************************************************************************%
%                          STORE CONFIG FUNCTION                          %
%*************************************************************************%
% This function is called by the RunGAMP callback function for the
% purpose of storing the current state/values of the relevant UI controls
% back to the GUIData object that will ultimately construct the Estim*,
% LinTrans, and GampOpt objects needed to run GAMP
function exitFlag = StoreConfig()
    % We must cycle through each tab, moving the values from the active UI
    % controls on each tab back to the GUIData object
    exitFlag = 0;   % Indicates successful storage
    
    % Migrate EstimIn tab values to GUIData object...
    Idx = data.EstimInUIData.selectedEstimIn;     % User-selected EstimIn class
    % Get handles to all UI controls on the active EstimIn card panel
    ChildHdls = get(gui.EstimInCards{Idx}, 'Children');
    for j = 1:numel(ChildHdls)
        % Identify the type of UI control associated with this handle.  If
        % it actually contains user-specified data, then copy that data to
        % the GUIData object
        switch lower( get(ChildHdls(j), 'Style') )
            case 'popupmenu'
                % Pop-up menu should contain an "allowable" field in the
                % GUIData EstimInUIData.Params field, so use UI value as
                % index into the appropriate "allowable" entry
                ParamIdx = strcmp(data.EstimInUIData.Params{Idx}(:,1), ...
                    get(ChildHdls(j), 'Tag'));  % Get index to param
                UIval = get(ChildHdls(j), 'Value'); % Selected pop-up entry
                val = data.EstimInUIData.Params{Idx}{ParamIdx,4}{UIval};   % Value of "allowable"
                data.EstimInUIData.Values{Idx}{ParamIdx,1} = val;    % Store in GUIData
            case 'edit'
                % Edit box should contain user-typed data.  Since we can
                % safely assume that this data will consist of 1 or more
                % numbers, we will treat it as such.
                ParamIdx = strcmp(data.EstimInUIData.Params{Idx}(:,1), ...
                    get(ChildHdls(j), 'Tag'));  % Get index to param
                StrVal = get(ChildHdls(j), 'String');   % User's entered string
                % Now try to convert the string into one or more numbers.
                % If this fails, report the fact to the user, and abort
                % remainder of script.
                try
                    val = eval(StrVal);
                    data.EstimInUIData.Values{Idx}{ParamIdx,1} = val;    % Store in GUIData
                catch ME
                    exitFlag = 1;   % Flag an error for calling fxn
                    msgStr = sprintf(['Failed to save EstimIn parameter ' ...
                        '%s successfully.  MATLAB error message: %s'], ...
                        data.EstimInUIData.Params{Idx}{ParamIdx,1}, ...
                        ME.message);
                    msgbox(msgStr, 'Error with user-provided input', 'error');
                    return
                end
            case 'checkbox'
                % This is an easy one.  Just find the relevant parameter
                % and copy the checkbox logical state over.
                ParamIdx = strcmp(data.EstimInUIData.Params{Idx}(:,1), ...
                    get(ChildHdls(j), 'Tag'));  % Get index to param
                data.EstimInUIData.Values{Idx}{ParamIdx,1} = ...
                    logical(get(ChildHdls(j), 'Value'));
            case 'listbox'
                % Not handling this one yet
            otherwise
                % Nothing to do for this parameter
                continue;
        end
    end
    
    % Migrate EstimOut tab values to GUIData object...
    Idx = data.EstimOutUIData.selectedEstimOut;     % User-selected EstimOut class
    % Get handles to all UI controls on the active EstimOut card panel
    ChildHdls = get(gui.EstimOutCards{Idx}, 'Children');
    for j = 1:numel(ChildHdls)
        % Identify the type of UI control associated with this handle.  If
        % it actually contains user-specified data, then copy that data to
        % the GUIData object
        switch lower( get(ChildHdls(j), 'Style') )
            case 'popupmenu'
                % Pop-up menu should contain an "allowable" field in the
                % GUIData EstimOutUIData.Params field, so use UI value as
                % index into the appropriate "allowable" entry
                ParamIdx = strcmp(data.EstimOutUIData.Params{Idx}(:,1), ...
                    get(ChildHdls(j), 'Tag'));  % Get index to param
                UIval = get(ChildHdls(j), 'Value'); % Selected pop-up entry
                val = data.EstimOutUIData.Params{Idx}{ParamIdx,4}{UIval};   % Value of "allowable"
                data.EstimOutUIData.Values{Idx}{ParamIdx,1} = val;    % Store in GUIData
            case 'edit'
                % Edit box should contain user-typed data.  Since we can
                % safely assume that this data will consist of 1 or more
                % numbers, we will treat it as such.
                ParamIdx = strcmp(data.EstimOutUIData.Params{Idx}(:,1), ...
                    get(ChildHdls(j), 'Tag'));  % Get index to param
                StrVal = get(ChildHdls(j), 'String');   % User's entered string
                % Now try to convert the string into one or more numbers.
                % If this fails, report the fact to the user, and abort
                % remainder of script.
                try
                    val = eval(StrVal);
                    data.EstimOutUIData.Values{Idx}{ParamIdx,1} = val;    % Store in GUIData
                catch ME
                    exitFlag = 1;   % Flag an error for calling fxn
                    msgStr = sprintf(['Failed to save EstimOut parameter ' ...
                        '%s successfully.  MATLAB error message: %s'], ...
                        data.EstimOutUIData.Params{Idx}{ParamIdx,1}, ...
                        ME.message);
                    msgbox(msgStr, 'Error with user-provided input', 'error');
                    return
                end
            case 'checkbox'
                % This is an easy one.  Just find the relevant parameter
                % and copy the checkbox logical state over.
                ParamIdx = strcmp(data.EstimOutUIData.Params{Idx}(:,1), ...
                    get(ChildHdls(j), 'Tag'));  % Get index to param
                data.EstimOutUIData.Values{Idx}{ParamIdx,1} = ...
                    logical(get(ChildHdls(j), 'Value'));
            case 'listbox'
                % Not handling this one yet
            otherwise
                % Nothing to do for this parameter
                continue;
        end
    end
    
    
    % Migrate Dataset tab values to GUIData object...
    % Get handles to all UI controls on the synthetic data-related panel
    ChildHdls = get(gui.SynthDataGrid, 'Children');
    for j = 1:numel(ChildHdls)
        % Identify the type of UI control associated with this handle.  If
        % it actually contains user-specified data, then copy that data to
        % the GUIData object
        switch lower( get(ChildHdls(j), 'Style') )
            case 'popupmenu'
                % Pop-up menu should contain an "allowable" field in the
                % GUIData GAMPOptUIData.Params field, so use UI value as
                % index into the appropriate "allowable" entry
                ParamIdx = strcmp(data.DatasetUIData.SynthParams(:,1), ...
                    get(ChildHdls(j), 'Tag'));  % Get index to param
                UIval = get(ChildHdls(j), 'Value'); % Selected pop-up entry
                val = data.DatasetUIData.SynthParams{ParamIdx,4}{UIval};   % Value of "allowable"
                data.DatasetUIData.SynthValues{ParamIdx,1} = val;    % Store in GUIData
            case 'edit'
                % Edit box should contain user-typed data.  Since we can
                % safely assume that this data will consist of 1 or more
                % numbers, we will treat it as such.
                ParamIdx = strcmp(data.DatasetUIData.SynthParams(:,1), ...
                    get(ChildHdls(j), 'Tag'));  % Get index to param
                StrVal = get(ChildHdls(j), 'String');   % User's entered string
                % Now try to convert the string into one or more numbers.
                % If this fails, report the fact to the user, and abort
                % remainder of script.
                try
                    val = eval(StrVal);
                    data.DatasetUIData.SynthValues{ParamIdx,1} = val;    % Store in GUIData
                catch ME
                    exitFlag = 1;   % Flag an error for calling fxn
                    msgStr = sprintf(['Failed to save Dataset parameter ' ...
                        '%s successfully.  MATLAB error message: %s'], ...
                        data.DatasetUIData.SynthParams{ParamIdx,1}, ...
                        ME.message);
                    msgbox(msgStr, 'Error with user-provided input', 'error');
                    return
                end
            case 'checkbox'
                % This is an easy one.  Just find the relevant parameter
                % and copy the checkbox logical state over.
                ParamIdx = strcmp(data.DatasetUIData.SynthParams(:,1), ...
                    get(ChildHdls(j), 'Tag'));  % Get index to param
                data.DatasetUIData.SynthValues{ParamIdx,1} = ...
                    logical(get(ChildHdls(j), 'Value'));
            case 'listbox'
                % Not handling this one yet
            otherwise
                % Nothing to do for this parameter
                continue;
        end
    end
    % Now, for the presently selected alternative EstimIn synthetic data
    % generation choice, copy its parameter names (tags) and values into a
    % cell array in the GUIData object
    Idx = data.DatasetUIData.selectedAltEstimIn;
    ChildHdls = get(gui.AltEstimInCards{Idx}, 'Children');
    % Clear existing contents of obj.DatasetUIData.AltEstimIn
    data.DatasetUIData.AltEstimIn = {};
    for j = 1:numel(ChildHdls)
        % Identify the type of UI control associated with this handle.  If
        % it actually contains user-specified data, then copy that data to
        % the GUIData object
        switch lower( get(ChildHdls(j), 'Style') )
            case 'popupmenu'
                % Pop-up menu should contain an "allowable" field in the
                % GUIData GAMPOptUIData.Params field, so use UI value as
                % index into the appropriate "allowable" entry
                ParamIdx = strcmp(data.EstimInUIData.Params{Idx}(:,1), ...
                    get(ChildHdls(j), 'Tag'));  % Get index to param
                UIval = get(ChildHdls(j), 'Value'); % Selected pop-up entry
                val = data.EstimInUIData.Params{Idx}{ParamIdx,4}{UIval};   % Value of "allowable"
                data.DatasetUIData.AltEstimIn = ...
                    [data.DatasetUIData.AltEstimIn();
                    {get(ChildHdls(j), 'Tag'), val}];
            case 'edit'
                % Edit box should contain user-typed data.  Since we can
                % safely assume that this data will consist of 1 or more
                % numbers, we will treat it as such.
                StrVal = get(ChildHdls(j), 'String');   % User's entered string
                % Now try to convert the string into one or more numbers.
                % If this fails, report the fact to the user, and abort
                % remainder of script.
                try
                    val = eval(StrVal);
                    data.DatasetUIData.AltEstimIn = ...
                        [data.DatasetUIData.AltEstimIn();
                        {get(ChildHdls(j), 'Tag'), val}];
                catch ME
                    exitFlag = 1;   % Flag an error for calling fxn
                    msgStr = sprintf(['Failed to save Dataset parameter ' ...
                        '%s successfully.  MATLAB error message: %s'], ...
                        get(ChildHdls(j), 'Tag'), ME.message);
                    msgbox(msgStr, 'Error with user-provided input', 'error');
                    return
                end
            case 'checkbox'
                % This is an easy one.  Just find the relevant parameter
                % and copy the checkbox logical state over.
                val = logical(get(ChildHdls(j), 'Value'));
                data.DatasetUIData.AltEstimIn = ...
                        [data.DatasetUIData.AltEstimIn();
                        {get(ChildHdls(j), 'Tag'), val}];
            otherwise
                % Nothing to do for this parameter
                continue;
        end
    end
    
    % Migrate Options tab values to GUIData object...
    % Get handles to all UI controls on the GAMP-related Options panel
    for i = 1:3
        ChildHdls = get(gui.GAMPOptionsGrid{i}, 'Children');
        for j = 1:numel(ChildHdls)
            % Identify the type of UI control associated with this handle.  If
            % it actually contains user-specified data, then copy that data to
            % the GUIData object
            switch lower( get(ChildHdls(j), 'Style') )
                case 'popupmenu'
                    % Pop-up menu should contain an "allowable" field in the
                    % GUIData GAMPOptUIData.Params field, so use UI value as
                    % index into the appropriate "allowable" entry
                    ParamIdx = strcmp(data.GAMPOptUIData.GAMPParams(:,1), ...
                        get(ChildHdls(j), 'Tag'));  % Get index to param
                    UIval = get(ChildHdls(j), 'Value'); % Selected pop-up entry
                    val = data.GAMPOptUIData.GAMPParams{ParamIdx,4}{UIval};   % Value of "allowable"
                    data.GAMPOptUIData.GAMPValues{ParamIdx,1} = val;    % Store in GUIData
                case 'edit'
                    % Edit box should contain user-typed data.  Since we can
                    % safely assume that this data will consist of 1 or more
                    % numbers, we will treat it as such.
                    ParamIdx = strcmp(data.GAMPOptUIData.GAMPParams(:,1), ...
                        get(ChildHdls(j), 'Tag'));  % Get index to param
                    StrVal = get(ChildHdls(j), 'String');   % User's entered string
                    % Now try to convert the string into one or more numbers.
                    % If this fails, report the fact to the user, and abort
                    % remainder of script.
                    try
                        val = eval(StrVal);
                        data.GAMPOptUIData.GAMPValues{ParamIdx,1} = val;    % Store in GUIData
                    catch ME
                        exitFlag = 1;   % Flag an error for calling fxn
                        msgStr = sprintf(['Failed to save Options parameter ' ...
                            '%s successfully.  MATLAB error message: %s'], ...
                            data.GAMPOptUIData.GAMPParams{ParamIdx,1}, ...
                            ME.message);
                        msgbox(msgStr, 'Error with user-provided input', 'error');
                        return
                    end
                case 'checkbox'
                    % This is an easy one.  Just find the relevant parameter
                    % and copy the checkbox logical state over.
                    ParamIdx = strcmp(data.GAMPOptUIData.GAMPParams(:,1), ...
                        get(ChildHdls(j), 'Tag'));  % Get index to param
                    data.GAMPOptUIData.GAMPValues{ParamIdx,1} = ...
                        logical(get(ChildHdls(j), 'Value'));
                case 'listbox'
                    % Not handling this one yet
                otherwise
                    % Nothing to do for this parameter
                    continue;
            end
        end
    end
end


%*************************************************************************%
%                    ON HELP MENU SELECTION FUNCTION                      %
%*************************************************************************%
function onHelp( ~, ~ )
    % User has asked for the documentation
    msg_str = ['Documentation on each of the individual EstimIn, ' ...
        'EstimOut, LinTrans, and GampOpt ' ...
        'classes, and the relevant parameters associated with each ' ...
        'class, can be found in the respective class definition files' ...
        ' (e.g., type "help GaussMixEstimOut" at the command line to view ' ...
        'documentation for the Gaussian-Mixture EstimOut class.' ...
        '  For questions on what different objects of this GUI are for, ' ...
        'hover the mouse pointer over the objects in question to view' ...
        ' a pop-up hint.'];
    msgbox(msg_str, 'GAMP GUI Help', 'help');
end % onHelp


%*************************************************************************%
%                  ON ESTIMIN POPUP SELECTION FUNCTION                    %
%*************************************************************************%
% When the user makes a EstimIn class selection on the main/execution tab,
% this callback will change the visible EstimIn card that is displayed on
% the EstimIn tab, and store in the GUIData object the currently selected
% tab
function onEstimInPopupSelection(src, ~)    
    Idx = get(src, 'Value');    % Index of selected EstimIn class
    if Idx == 1
        % The default class was chosen
        data.EstimInUIData.selectedEstimIn = Idx;
        gui.EstimInCardPanel.SelectedChild = Idx;
    else
        % Because the pop-up menu contains one additional option than there
        % are actual EstimIn classes (the default), decrement the index by
        % one
        data.EstimInUIData.selectedEstimIn = Idx - 1;
        gui.EstimInCardPanel.SelectedChild = Idx - 1;
    end
end


%*************************************************************************%
%                  ON ESTIMOUT POPUP SELECTION FUNCTION                   %
%*************************************************************************%
% When the user makes an EstimOut class selection on the main/execution tab,
% this callback will change the visible EstimOut card that is displayed on
% the EstimOut tab, and store in the GUIData object the currently selected
% tab
function onEstimOutPopupSelection(src, ~)    
    Idx = get(src, 'Value');    % Index of selected EstimOut class
    if Idx == 1
        % The default class was chosen
        data.EstimOutUIData.selectedEstimOut = Idx;
        gui.EstimOutCardPanel.SelectedChild = Idx;
    else
        % Because the pop-up menu contains one additional option than there
        % are actual EstimOut classes (the default), decrement the index by
        % one
        data.EstimOutUIData.selectedEstimOut = Idx - 1;
        gui.EstimOutCardPanel.SelectedChild = Idx - 1;
    end
end


%*************************************************************************%
%                ON SPARSITY CHECKBOX SELECTION FUNCTION                  %
%*************************************************************************%
% When the user checks the "sparsify the EstimIn" checkbox, update some
% stuff here
function onSparsifyCheckboxSelection(src, ~)
    val = get(src, 'Value');
    if val == 1
        % Enable sparsification
        data.EstimInUIData.Sparsify = true;     % Set flag
        set(gui.SparsityBar, 'Visible', 'on');  % Show sparsity rate
    else
        % Disable sparsification
        data.EstimInUIData.Sparsify = false;    % Clear flag
        set(gui.SparsityBar, 'Visible', 'off');	% Hide sparsity rate
    end
end

%*************************************************************************%
%                ON SPARSITY RATE EDIT ACTIVITY FUNCTION                  %
%*************************************************************************%
% When the user changes the Bernoulli-* sparsity rate, push the changes to
% the GUIData object
function onSparsityRateEditActivity(src, ~)
    try
        StrVal = get(src, 'String');    % Get user input
        val = str2double(StrVal);       % Try to convert to scalar
        if isnumeric(val) && isscalar(val) && val >= 0 && val <= 1
            data.EstimInUIData.SparsityRate = val;    % Store in GUIData
        else
            msgbox('Please enter a scalar in the interval [0,1]', ...
                'Error with user-provided input', 'error');
            set(src, 'String', num2str(data.EstimInUIData.SparsityRate));
        end
    catch ME
        exitFlag = 1;   % Flag an error for calling fxn
        msgStr = sprintf(['Failed to update sparsity rate ' ...
            'successfully.  MATLAB error message: %s'], ME.message);
        msgbox(msgStr, 'Error with user-provided input', 'error');
        return
    end
end


%*************************************************************************%
%                ON ALT SPARSITY CHECKBOX SELECTION FUNCTION              %
%*************************************************************************%
% When the user checks the "sparsify the EstimIn" checkbox, update some
% stuff here FOR THE DATASET TAB
function onAltSparsifyCheckboxSelection(src, ~)
    val = get(src, 'Value');
    if val == 1
        % Enable sparsification
        data.DatasetUIData.Sparsify = true;     % Set flag
        set(gui.AltSparsityBar, 'Visible', 'on');  % Show sparsity rate
    else
        % Disable sparsification
        data.DatasetUIData.Sparsify = false;    % Clear flag
        set(gui.AltSparsityBar, 'Visible', 'off');	% Hide sparsity rate
    end
end

%*************************************************************************%
%                ON ALT SPARSITY RATE EDIT ACTIVITY FUNCTION              %
%*************************************************************************%
% When the user changes the Bernoulli-* sparsity rate, push the changes to
% the GUIData object FOR THE DATASET TAB
function onAltSparsityRateEditActivity(src, ~)
    try
        StrVal = get(src, 'String');    % Get user input
        val = str2double(StrVal);       % Try to convert to scalar
        if isnumeric(val) && isscalar(val) && val >= 0 && val <= 1
            data.DatasetUIData.SparsityRate = val;    % Store in GUIData
        else
            msgbox('Please enter a scalar in the interval [0,1]', ...
                'Error with user-provided input', 'error');
            set(src, 'String', num2str(data.DatasetUIData.SparsityRate));
        end
    catch ME
        exitFlag = 1;   % Flag an error for calling fxn
        msgStr = sprintf(['Failed to update sparsity rate ' ...
            'successfully.  MATLAB error message: %s'], ME.message);
        msgbox(msgStr, 'Error with user-provided input', 'error');
        return
    end
end


%*************************************************************************%
%             ON SYNTHETIC DATASET RADIO SELECTION FUNCTION               %
%*************************************************************************%
% If user selected to generate a synthetic dataset, then disable all
% workspace dataset-related UI controls, negate the workspace dataset radio
% button, and save state to GUIData object
function onSynthDataSelection(src, ~)
    % Toggle the opposite value for workspace dataset radio button
    set(gui.WSpcRadioBtn, 'Value', not( get(src, 'Value') ));
    
    % Store state to GUIData
    if get(src, 'Value')
        data.DatasetUIData.GenSynthData = true;
        % Disable all workspace dataset panel UI controls
        set(gui.SynthPanel, 'Enable', 'on');
        set(gui.WSpcPanel, 'Enable', 'off');
    else
        data.DatasetUIData.GenSynthData = false;
        % Enable all workspace dataset panel UI controls
        set(gui.SynthPanel, 'Enable', 'off');
        set(gui.WSpcPanel, 'Enable', 'on');
    end
    
end


%*************************************************************************%
%             ON WORKSPACE DATASET RADIO SELECTION FUNCTION               %
%*************************************************************************%
% If user selected to generate a workspace dataset, then disable all
% synthetic dataset-related UI controls, negate the synthetic dataset radio
% button, and save state to GUIData object
function onWSpcDataSelection(src, ~)
    % Toggle the opposite value for workspace dataset radio button
    set(gui.SynthRadioBtn, 'Value', not( get(src, 'Value') ));
    
    % Store state to GUIData
    if get(src, 'Value')
        data.DatasetUIData.GenSynthData = false;
        % Disable all synthetic dataset panel UI controls
        set(gui.SynthPanel, 'Enable', 'off');
        set(gui.WSpcPanel, 'Enable', 'on');
    else
        data.DatasetUIData.GenSynthData = true;
        % Enable all synthetic dataset panel UI controls
        set(gui.SynthPanel, 'Enable', 'on');
        set(gui.WSpcPanel, 'Enable', 'off');
    end
    
end


%*************************************************************************%
%                  ON REFRESH LIST SELECTION FUNCTION                     %
%*************************************************************************%
% If the user presses this button in the Dataset tab, it will refresh the
% lists of user workspace variable names
function onRefreshListSelection(src, ~)
    % Get workspace variable names
    var_names = evalin('base', 'who');
    var_names = [{[]}; var_names];
    
    % Store names in listboxes
    set(gui.AListBox, 'String', var_names);
    onListBoxSelection(gui.AListBox);
    set(gui.yListBox, 'String', var_names);
    onListBoxSelection(gui.yListBox);
    set(gui.xtrueListBox, 'String', var_names);
    onListBoxSelection(gui.xtrueListBox);
end


%*************************************************************************%
%                    ON LIST BOX SELECTION FUNCTION                       %
%*************************************************************************%
% If the user intends to select training and test datasets from variables
% in their workspace, this function will be called whenever they make a
% selection from workspace variable lists in the Dataset tab.  This script
% will store the selected variable in the GUIData object, data.
function onListBoxSelection(src, ~)
    % Get the value of the selected variable from the user's workspace
    Idx = get(src, 'Value');            % Selection index
    var_names = get(src, 'String');     % List of all variables in listbox
    if isempty(Idx) || (Idx > numel(var_names))
        % List of elements has shortened
        set(src, 'Value', 1);
        value = NaN;
    else
        chosen_var_name = var_names{Idx};   % Name of chosen workspace variable
        if isempty(chosen_var_name)
            return
        end
        value = evalin('base', chosen_var_name);    % Value of variable
    end
    
    % Store the variable in the appropriate location in the GUIData object
    switch get(src, 'Tag')
        case 'A'
            % Forward operator
            if ~isa(value, 'LinTrans')
                if isnumeric(value)
                    % A is a matrix, so convert to LinTrans
                    value = MatrixLinTrans(value);
                else
                    msgbox(sprintf('%s is not a valid LinTrans object', ...
                        chosen_var_name), 'Unrecognized LinTrans class', ...
                        'error');
                end
            end
            data.DatasetUIData.WSpcValues{1,1} = value;
        case 'y'
            % Observations
            if ~isnumeric(value)
                msgbox(sprintf('%s should be a numeric array', ...
                    chosen_var_name), 'Non-numeric observations', ...
                    'error');
            end
            data.DatasetUIData.WSpcValues{2,1} = value;
        case 'x_true'
            % Ground truth
            if ~isnumeric(value)
                msgbox(sprintf('%s should be a numeric array', ...
                    chosen_var_name), 'Non-numeric observations', ...
                    'error');
            end
            data.DatasetUIData.WSpcValues{3,1} = value;
    end    
end


%*************************************************************************%
%                       ON TAB SELECTION FUNCTION                         %
%*************************************************************************%
% Whenever the Dataset tab is selected, refresh the listboxes that contain
% the names of variables in the user's workspace
function onTabSelection(~, eventData)
    if eventData.('SelectedChild') == 2
        onRefreshListSelection();
    end
end


%*************************************************************************%
%                          ON RUN GAMP FUNCTION                           %
%*************************************************************************%
% User pressed button to run the classifier(s), so do that now
function onRunGAMP(~, ~)
    
    % Copy all relevant state of the UI back to the GUIData object
    exitFlag = StoreConfig();
    if exitFlag ~= 0
        % Something failed with the data storage.  Abort script.
        return;
    end
    
    % Next, get training and test matrices, A_train and A_test, as well as
    % training and test binary labels, y_train and y_test
    [A, y, x_true, exitFlag] = data.generateDataset();
    if exitFlag ~= 0
        % Something failed with the dataset generation.  Abort script.
        return;
    end
    
    % Now, run GAMP on the provided dataset
    tic;
    [x_gamp, exitFlag] = RunGAMP(data, y, A);
    gamp_runtime = toc;
    if exitFlag ~= 0
        % Something failed with GAMP execution.  Abort script.
        return;
    end
    
    % Plot recovery and, if applicable, ground truth
    cla(gui.ViewAxes);      % Clear current axes
    if isreal(x_gamp)
        stem(gui.ViewAxes, x_gamp, 'r');
        legend_string = {'x_{gamp}'};
    else
        stem(gui.ViewAxes, abs(x_gamp), 'r');
        legend_string = {'|x_{gamp}|'};
    end
    if ~any(isnan(x_true(:)))
        % Ground truth exists
        hold(gui.ViewAxes, 'on');
        if isreal(x_true)
            stem(gui.ViewAxes, x_true, 'b')
            legend_string = [legend_string, 'x_{true}'];
        else
            stem(gui.ViewAxes, abs(x_true), 'b')
            legend_string = [legend_string, '|x_{true}|'];
        end
        hold(gui.ViewAxes, 'off');
    end
    legend(gui.ViewAxes, legend_string, 'Location', 'Best');
    
    % Calculate and report recovery statistics
    MSE = norm(x_gamp - x_true)^2;  % Mean-square error
    NMSE = MSE / norm(x_true)^2;
    NMSEdB = 10*log10(NMSE);        % Normalized mean-square error (dB)
    [M, N] = A.size();
    K = [sum(x_true ~= 0), sum(x_gamp ~= 0)];
    TrueMSE = [0; NaN];
    Stats = [
        TrueMSE(any(isnan(x_true(:)))+1), NMSEdB;
        TrueMSE(any(isnan(x_true(:)))+1), MSE;
        N, N;
        M, M;
        K;
        NaN, gamp_runtime;
        ];
    set(gui.StatTable, 'Data', Stats);
    
    % Output important variables to user workspace
    if get(gui.saveData, 'Value')
        assignin('base', 'A', A);
        assignin('base', 'y', y);
        if ~isnan(x_true), assignin('base', 'x_true', x_true); end
        assignin('base', 'x_gamp', x_gamp);
    end
end


%*************************************************************************%
%                   ON CLEAR BUTTON SELECTION FUNCTION                    %
%*************************************************************************%
% User pressed button to clear the axes and statistics
function onClear(~, ~)
    cla(gui.ViewAxes);      % Clear current axes
    set(gui.StatTable, 'Data', []);
end


%*************************************************************************%
%                   ON RESET BUTTON SELECTION FUNCTION                    %
%*************************************************************************%
% User pressed button to clear the axes and statistics
function onReset(~, ~)
    onClear();      % Clear plot and statistics table
    
    % Reset turboGAMP configuration
    set(gui.EstimInPopup, 'Value', 1);
    onEstimInPopupSelection(gui.EstimInPopup);
    set(gui.EstimOutPopup, 'Value', 1);
    onEstimOutPopupSelection(gui.EstimOutPopup);
    
    % Delete all existing UI controls from all tabs
    names = {'EstimInPanel', 'EstimOutPanel', 'DatasetPanel', 'OptionsPanel'};
    for j = 1:4
        ChildHdls = get(gui.(names{j}), 'Children');
        for i = 1:numel(ChildHdls)
            delete(ChildHdls(i));
        end
    end
    
    % Use timer to re-populate remaining tabs without causing lag when GUI
    % execute/main tab becomes visible
    startuptimer = timer('ExecutionMode', 'SingleShot', 'TimerFcn', ...
        @populateTabs, 'StartDelay', .5);
    start(startuptimer);
    while strcmp(get(startuptimer, 'Running'), 'on')
        pause(.1);
    end
    delete(startuptimer);
end


%*************************************************************************%
%                      ON NON ESTIMIN GEN FUNCTION                        %
%*************************************************************************%
% User pressed checkbox to enable generating the true x from an EstimIn
% class other than the one chosen for recovery, so enable the panel
function onNonEstimInGen(src, ~)
    if get(src, 'Value')
        % Box is checked
        val = 'on';
    else
        val = 'off';
    end
    set(gui.AltEstimInPanel, 'Enable', val);
end


%*************************************************************************%
%                 ON ALT ESTIMIN POPUP SELECTION FUNCTION                 %
%*************************************************************************%
% When the user makes an alternative EstimIn class selection on the 
% dataset-synthetic tab, this callback will change the visible EstimIn card 
% that is displayed on the AltEstimIn tab, and store in the GUIData object 
% the currently selected tab
function onAltEstimInPopupSelection(src, ~)    
    Idx = get(src, 'Value');    % Index of selected EstimIn class
    if Idx == 1
        % The default class was chosen
        data.DatasetUIData.selectedAltEstimIn = Idx;
        gui.AltEstimInCardPanel.SelectedChild = Idx;
    else
        % Because the pop-up menu contains one additional option than there
        % are actual EstimIn classes (the default), decrement the index by
        % one
        data.DatasetUIData.selectedAltEstimIn = Idx - 1;
        gui.AltEstimInCardPanel.SelectedChild = Idx - 1;
    end
end


%-------------------------------------------------------------------------%
function onExit( ~, ~ )
    % User wants to quit the application
    delete( gui.Window );
end % onExit

end % EOF