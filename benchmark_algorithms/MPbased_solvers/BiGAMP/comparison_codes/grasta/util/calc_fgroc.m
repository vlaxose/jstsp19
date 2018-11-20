function roc = calc_fgroc( video_path, frames_vec, frame_names_gt, fgs_cmp, MAX_gt )
%CALC_FGROC Summary of this function goes here
%   Detailed explanation goes here

threshvec=linspace(0,1,100);
roc=zeros(length(threshvec),2);
tidx_tot = 0; N=0;


% Vectors to keep false positives, false negatives, and true positives
fp_tot = zeros(1,length(threshvec));
fn_tot = zeros(1,length(threshvec));
tp_tot = zeros(1,length(threshvec));

% Go through all test frames
MAX_gt = min(length(frames_vec), MAX_gt);
for ff=1:MAX_gt,
    
    fname = [video_path filesep frame_names_gt{ff}];
    
    % prepare the image
    I = imread(fname);
    I = double(rgb2gray(I));
    truth_idx = find(I==255);  % Compare this index set to the sparse component.
    [m n]=size(I);
    N=N+m*n;
    tidx_tot=tidx_tot+length(truth_idx);
    fg_img = abs(fgs_cmp(:,ff));
    fg_img = fg_img/max(fg_img);% Put actual grasta sparse part here    
    
    
    for t=1:length(threshvec)
        detected_idx = find(fg_img>threshvec(t));
        
        false_pos = setdiff(detected_idx, truth_idx);
        false_neg = setdiff(truth_idx, detected_idx);
        true_pos = intersect(truth_idx, detected_idx);
        fp_tot(t) = fp_tot(t)+length(false_pos);
        fn_tot(t) = fn_tot(t)+length(false_neg);
        tp_tot(t) = tp_tot(t)+length(true_pos);
        
        
    end
    
end

for t=1:length(threshvec)

    roc(t,1)=fp_tot(t)/(N-tidx_tot); % false positive 
    roc(t,2)=tp_tot(t)/tidx_tot;    % recall = true positive  tp/(tp+fn)
end

end

