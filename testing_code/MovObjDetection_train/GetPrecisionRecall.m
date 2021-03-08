function [precision, recall, falsealarm, min_x_tp, min_y_tp, max_x_tp, max_y_tp, min_x_fp, min_y_fp, max_x_fp, max_y_fp, mean_x_tp, mean_y_tp, mean_x_fp, mean_y_fp] = GetPrecisionRecall(RefinedDetections,Groundtruth)
%GETPRECISIONRECALL Summary of this function goes here
%   Detailed explanation goes here
TruePositiveIdx = [];
dis = RefinedDetections(:,3:4);
RefinedDetections = RefinedDetections(:, 1:2);
disp_x = [];
if ~isempty(RefinedDetections)
    [idx1, dist1] = knnsearch(RefinedDetections, Groundtruth, 'K', 5);
    dist_idx1 = dist1 <= 10;
    dist_logical1 = zeros(size(dist1,1), 5);
    dist_logical1(dist_idx1) = idx1(dist_idx1);
    fn = sum(dist_logical1(:,1) == 0);
    
    GroundtruthUsage = false(size(Groundtruth, 1), 1);
    [idx2, dist2] = knnsearch(Groundtruth, RefinedDetections, 'K', 5);
    dist_idx2 = dist2 <= 10;
    dist_logical2 = zeros(size(dist2,1), 5);
    dist_logical2(dist_idx2) = idx2(dist_idx2);
    tp = 0;
    for j = 1:size(dist2, 1)
        for k = 1:5
            assignedIdx = dist_logical2(j, k);
            if assignedIdx == 0
                break;
            elseif assignedIdx > 0 && ~GroundtruthUsage(assignedIdx)
                tp = tp + 1;
                GroundtruthUsage(assignedIdx) = true;
                TruePositiveIdx = [TruePositiveIdx; j];
                break;
            end
        end
    end
    
    
    FalsePositiveIdx = [];
    g2 = 1;
    for g=1:size(TruePositiveIdx)
        if g2==TruePositiveIdx(g,1)
            g2 = g2+1;
        else
            FalsePositiveIdx = [FalsePositiveIdx; g2];
            g2 = g2+2;
        end
    end
    
    fp = size(RefinedDetections,1) - tp;
    
    disp_tp = dis(TruePositiveIdx,:);
    disp_fp = dis(FalsePositiveIdx, :);
    
    if length(TruePositiveIdx) > 0   
        min_x_tp = min(abs(disp_tp(:,1)));
        min_y_tp = min(abs(disp_tp(:,2)));
        max_x_tp = max(abs(disp_tp(:,1)));
        max_y_tp = max(abs(disp_tp(:,2)));
        mean_x_tp = mean(abs(disp_tp(:,1)));
        mean_y_tp = mean(abs(disp_tp(:,2)));
    else
        min_x_tp=100; min_y_tp=100; max_x_tp=0; max_y_tp=0; mean_x_tp=0; mean_y_tp=0;
    end
    
    if length(FalsePositiveIdx) > 0
        min_x_fp = min(abs(disp_fp(:,1)));
        min_y_fp = min(abs(disp_fp(:,2)));
        max_x_fp = max(abs(disp_fp(:,1)));
        max_y_fp = max(abs(disp_fp(:,2)));
        mean_x_fp = mean(abs(disp_fp(:,1)));
        mean_y_fp = mean(abs(disp_fp(:,2)));
    else
        min_x_fp=100; min_y_fp=100; max_x_fp=0; max_y_fp=0; mean_x_fp=0; mean_y_fp=0;
    end
    
    precision = tp/(tp+fp);
    recall = tp/(tp+fn);
    falsealarm = RefinedDetections(setdiff(1:size(RefinedDetections, 1), TruePositiveIdx), :);
end

