close all; clear all; clc;

annot_files = dir(fullfile('/home/casper/Desktop/MovingObjDetector-WAMI.matlab/AOI/AOI_03/annotations_moving/*.txt'));
cet_files = dir(fullfile('/home/casper/Desktop/MovingObjDetector-WAMI.matlab/AOI/AOI_03/ct_locations_prof_displacements/*.txt'));

precisions_2 = [];
recalls_2 = [];
f1_scores_2 = [];

min_x_tps = [];
min_y_tps = [];
max_x_tps = [];
max_y_tps = [];
min_x_fps = [];
min_y_fps = [];
max_x_fps = [];
max_y_fps = [];
mean_x_tps = [];
mean_y_tps = [];
mean_x_fps = [];
mean_y_fps = [];

i = 1;
j = 1;
cnt = 1;
while i<=length(cet_files)
    gt_file = [annot_files(j).folder '/' annot_files(j).name];
    Gt = csvread(gt_file);
    Groundtruth = Gt(:,1:2);
    disp(annot_files(j).name);
    disp(cet_files(i).name);
    if strcmp(annot_files(j).name, cet_files(i).name)
        dt_file = [cet_files(i).folder '/' cet_files(i).name];
        dt = csvread(dt_file);
        RefinedDetections = dt;
        [precision, recall, falsealarm, min_x_tp, min_y_tp, max_x_tp, max_y_tp, min_x_fp, min_y_fp, max_x_fp, max_y_fp, mean_x_tp, mean_y_tp, mean_x_fp, mean_y_fp] = GetPrecisionRecall(RefinedDetections,Groundtruth);
        precisions_2(i) = precision;
        recalls_2(i) = recall;
        if precision > 0 ||  recall > 0
            f1_scores_2(cnt) = (2*precision*recall)/(precision+recall);
            cnt = cnt+1;
        end
        min_x_tps = [min_x_tps; min_x_tp];
        min_y_tps = [min_y_tps; min_y_tp];
        max_x_tps = [max_x_tps; max_x_tp];
        max_y_tps = [max_y_tps; max_y_tp];
        min_x_fps = [min_x_fps; min_x_fp];
        min_y_fps = [min_y_fps; min_y_fp];
        max_x_fps = [max_x_fps; max_x_fp];
        max_y_fps = [max_y_fps; max_y_fp];
        mean_x_tps = [mean_x_tps; mean_x_tp];
        mean_y_tps = [mean_y_tps; mean_y_tp];
        mean_x_fps = [mean_x_fps; mean_x_fp];
        mean_y_fps = [mean_y_fps; mean_y_fp];
        i = i+1;
        j = j+1;
    else
        j = j+1;
    end        
end

avg_precision = mean(precisions_2);
avg_recall = mean(recalls_2);
avg_f1_score = mean(f1_scores_2);
disp(["Average Precision: " avg_precision]);
disp(["Average Recall: " avg_recall]);
disp(["Average F1 score: " avg_f1_score]);
disp("")
disp(["Min_x_tp: " min(min_x_tps)]);
disp(["Min_y_tp: " min(min_y_tps)]);
disp(["Max_x_tp: " max(max_x_tps)]);
disp(["Max_y_tp: " max(max_y_tps)]);
disp("")
disp(["Min_x_fp: " min(min_x_fps)]);
disp(["Min_y_fp: " min(min_y_fps)]);
disp(["Max_x_fp: " max(max_x_fps)]);
disp(["Max_y_fp: " max(max_y_fps)]);
disp("")
disp(["Mean_x_tp: " mean(mean_x_tps)]);
disp(["Mean_y_tp: " mean(mean_y_tps)]);
disp(["Mean_x_fp: " mean(mean_x_fps)]);
disp(["Mean_y_fp: " mean(mean_y_fps)]);