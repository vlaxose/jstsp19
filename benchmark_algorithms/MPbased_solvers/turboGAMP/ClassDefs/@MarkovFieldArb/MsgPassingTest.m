N = 7;    % Dimension of X
beta = 0.4;
alpha = 0;

PI_OUT = rand(N,1);

B = double(rand(N) < 1/3);
B = B'*B;
B = B .* ~eye(N);
B = (B ~= 0);
maxDeg = max(sum(B,2));

dumID = N+1;

NbrList = dumID * ones(N,maxDeg);
for i = 1:N
Idxs = find(B(i,:) == 1);
NbrList(i,1:numel(Idxs)) = Idxs;
end


% NbrList is a bookkeeping array whose i^th row tells us the 
% indices of the destination nodes for messages leaving node i.
% RevNbrList is a linearly-indexed mapping that takes the
% outgoing messages from the outgoing message array and places
% them in the correct places in the incoming message array for
% the next message passing iteration
dumIdx = find(NbrList == dummyID, 1); 	% Index of 1st dummy loc
revNbrList = dumIdx * ones(N,maxDeg);   % Default to dummy outgoing msgs
OffsetAdjMtx = spalloc(N, N, N*maxDeg);
for i = 1:N
    OffsetAdjMtx(:,i) = sum(Mtx(:,1:i), 2) .* Mtx(:,i);
    rowIdcs = NbrList(i,1:InDeg(i));
    revNbrList(i,1:InDeg(i)) = ...
        N*(OffsetAdjMtx(rowIdcs,i) - 1) + rowIdcs';
end
dummyMask = (NbrList == dummyID);   % 1's in dummy node locs


LabelArr = ceil(3*rand(N,1)); 	% Graph coloring label array
UniqLbls = unique(LabelArr);    % Unique labels
C = numel(UniqLbls);         	% Total number of unique labels

% Declare model parameter-related constants        
eb = exp(beta);
ebi = exp(-beta);
e0 = exp(alpha);
e1 = exp(-alpha);

% Initialize incoming probability messages
nodePot = NaN(N,1,2);
nodePot(:,1,1) = 1 - PI_OUT;       	% Prob. that node is 0
nodePot(:,1,2) = PI_OUT;            % Prob. that node is 1

% Initialize incoming and outgoing messages.  Both inbound and
% outbound messages will be stored as 2-element cell arrays.
% The following description applies to inbound message storage:
% The first cell contains a sparse N-by-N+1 matrix, with the
% j^th row containing the inbound s(j) = 0 messages from all
% other nodes (including an (N+1)^th "dummy" node that is used 
% purely for computational convenience).  Since only a handful
% of other nodes are connected to node j, we only store the
% entries in the sparse matrix corresponding to actual
% neighbors.  Similarly, the second cell contains s(j) = 1
% messages.
%             InboundMsgs = cell(2,1);
%             InboundMsgs{1} = sparse(N,N+1); InboundMsgs{2} = sparse(N,N+1);
%             InboundMsgs{1}(NbrListLin) = 1/2;   % Init. inbound s = 0 msgs
%             InboundMsgs{2}(NbrListLin) = 1/2;   % Init. inbound s = 1 msgs
%             OutboundMsgs = cell(2,1);
%             OutboundMsgs{1} = sparse(N,N+1); OutboundMsgs{2} = sparse(N,N+1);
%             OutboundMsgs{1}(NbrListLin) = 1/2;	% Init. outbound s = 0 msgs
%             OutboundMsgs{2}(NbrListLin) = 1/2;	% Init. outbound s = 1 msgs
InboundMsgs = cell(2,1);
InboundMsgs{1} = 1/2*ones(N,maxDeg);    % Init. inbound s = 0 msgs
InboundMsgs{2} = 1/2*ones(N,maxDeg);    % Init. inbound s = 1 msgs
OutboundMsgs = cell(2,1);
OutboundMsgs{1} = 1/2*ones(N,maxDeg);	% Init. outbound s = 0 msgs
OutboundMsgs{2} = 1/2*ones(N,maxDeg);	% Init. outbound s = 1 msgs

% Initialize incoming and outgoing messages.  Both inbound and
% outbound messages will be stored as 2-element cell arrays.
SpatialMsgsIn = cell(2,1);
SpatialMsgsIn{1} = 1/2*ones(N,maxDeg); 	% Init. spatial inbound s = 0 msgs
SpatialMsgsIn{2} = 1/2*ones(N,maxDeg);  	% Init. spatial inbound s = 1 msgs
SpatialMsgsOut = cell(2,1);
SpatialMsgsOut{1} = 1/2*ones(N,maxDeg);  	% Init. spatial outbound s = 0 msgs
SpatialMsgsOut{2} = 1/2*ones(N,maxDeg);  	% Init. spatial outbound s = 1 msgs

prod0 = zeros(Nx,Ny,Nz);
prod1 = prod0;

for iter = 1:C*obj.maxIter
    % First grab indices of nodes to update this round
    UpdNodes = (LabelArr == UniqLbls(mod(iter-1,C)+1));
    NumUpd = sum(UpdNodes);
    SubIdx = [1:N]'; SubIdx = SubIdx(UpdNodes);

    % Now cycle through each outbound message being dispatched
    % from a given node (many will be outbound to the dummy
    % node, and are therefore irrelevant and will be
    % overwritten)
    for out = 1:maxDeg
        InclIdx = [1:out-1, out+1:maxDeg];  % Neighbor columns to include
        prod0 = e0 * nodePot(UpdNodes,1,1) .* ...
            prod(InboundMsgs{1}(UpdNodes,InclIdx), 2);
        prod1 = e1 * nodePot(UpdNodes,1,2) .* ...
            prod(InboundMsgs{2}(UpdNodes,InclIdx), 2);
        p0 = prod0*eb + prod1*ebi;
        p1 = prod0*ebi + prod1*eb;
        sump0p1 = p0+p1;

        OutboundMsgs{1}(UpdNodes,out) = p0 ./ sump0p1;
        OutboundMsgs{2}(UpdNodes,out) = p1 ./ sump0p1;
    end

    % Now copy the outbound messages over to the inbound
    % messages for the next message passing cycle
    NbrListMask = (NbrList ~= dumID);
    NbrListMask(~UpdNodes,:) = 0;           % Wipe out irrelevant nodes
    NbrListTrunc = NbrList(UpdNodes,:);
    NbrListTruncMask = (NbrListTrunc ~= dumID);
    NbrListLin = reshape(sub2ind([N+1, N+1], ...
        repmat(SubIdx, maxDeg, 1), NbrListTrunc), N, NumUpd);
    MsgXfer = cell(2,1);            % Sparse matrix for temporary storage
    MsgXfer{1} = sparse(N+1,N+1); MsgXfer{2} = sparse(N+1,N+1);
    MsgXfer{1}(NbrListLin) = OutboundMsgs{1}(UpdNodes,:);
    MsgXfer{2}(NbrListLin) = OutboundMsgs{2}(UpdNodes,:);
    MsgXfer{1} = MsgXfer{1}(1:N,1:N)';
    MsgXfer{2} = MsgXfer{2}(1:N,1:N)';
    NbrListTrunc2 = NbrListTrunc;
    NbrListTrunc2(~NbrListTruncMask) = 1;
    NbrListRed = reshape(sub2ind([N, N], ...
        repmat(SubIdx, maxDeg, 1), NbrListTrunc2(:)), N, NumUpd);
    InboundMsgs{1}(UpdNodes,:) = 1/2;
    InboundMsgs{2}(UpdNodes,:) = 1/2;
    InboundMsgs{1}(NbrListMask) = MsgXfer{1}(NbrListRed(NbrListTruncMask));
    InboundMsgs{2}(NbrListMask) = MsgXfer{2}(NbrListRed(NbrListTruncMask));
    
    
    
    % ***** METHOD B ****
    % *************** Spatial message updates *****************
    % Now cycle through each outbound message being dispatched
    % from a given node (many will be outbound to the dummy
    % node, and are therefore irrelevant and will be
    % overwritten)
    SpatialMsgsOut{1} = SpatialMsgsIn{1}(revNbrList);
    SpatialMsgsOut{2} = SpatialMsgsIn{2}(revNbrList);
    for out = 1:maxDeg
        InclIdx = [1:out-1, out+1:maxDeg];  % Neighbor columns to include
        prod0 = e0 * nodePot{1}(UpdNodes,1) .* ...
            prod(SpatialMsgsIn{1}(UpdNodes,InclIdx), 2) .* ...
            PrevTimeMsgsIn{1}(UpdNodes,1) .* ...
            LateTimeMsgsIn{1}(UpdNodes,1);
        prod1 = e1 * nodePot{2}(UpdNodes,1) .* ...
            prod(SpatialMsgsIn{2}(UpdNodes,InclIdx), 2) .* ...
            PrevTimeMsgsIn{2}(UpdNodes,1) .* ...
            LateTimeMsgsIn{2}(UpdNodes,1);
        p0 = prod0*eb + prod1*ebi;
        p1 = prod0*ebi + prod1*eb;
        sump0p1 = p0+p1;
        
        SpatialMsgsOut{1}(UpdNodes,out) = p0 ./ sump0p1;
        SpatialMsgsOut{2}(UpdNodes,out) = p1 ./ sump0p1;
    end
    % Replace outgoing messages to dummy nodes with
    % uninformative values
    SpatialMsgsOut{1}(dummyMask) = 1/2;
    SpatialMsgsOut{2}(dummyMask) = 1/2;
    
    % Now copy the outbound messages over to the inbound
    % messages for the next message passing cycle
    SpatialMsgsIn{1} = SpatialMsgsOut{1}(revNbrList);
    SpatialMsgsIn{2} = SpatialMsgsOut{2}(revNbrList);
    
    
    fprintf('Update difference: %f\n', norm(InboundMsgs{1} - SpatialMsgsIn{1}, 'fro'))
end


% Compute extrinsic likelihood, marginal potential and s_hat
msgProds = NaN(N,1,2);
msgProds(:,1,1) = prod(InboundMsgs{1}, 2)*e0;
msgProds(:,1,2) = prod(InboundMsgs{2}, 2)*e1;
Le_spdec = log(msgProds(:,1,2)./msgProds(:,1,1));
PI_IN = 1 ./ (1 + exp(-Le_spdec));

msgProds = msgProds.*nodePot;
sumMsgProds = sum(msgProds, 3);
S_POST = msgProds(:,1,2) ./ sumMsgProds;    % Pr{S(n) = 1 | Y}
S_HAT = double(S_POST > 1/2);