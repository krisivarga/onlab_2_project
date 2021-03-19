% run_fuse_MVKD_B
%
% B - session 2 for suspect, session 1 for offender

clear all
addpath('.\m_files', '.\m_files\cllr\', '.\m_files\fusion\');

complete_separation = true;

% data sets
data_set_prefix = 'MVKD2';
%                1    2    3    4    5    6    
vowel_labels = {'aI' 'eI' 'oU' 'aU' 'OI' 'all'};
which_vowel = 1:5;

num_vowels = length(which_vowel);

% load data
log_scores_cell = cell(1, num_vowels);
II_pairs_cell = cell(1, num_vowels);
log_scores_train_LogReg_ss_cell = cell(1, num_vowels);
log_scores_train_LogReg_ds_cell = cell(1, num_vowels);
for I_vowel = which_vowel
    load(['.\resultsB\', data_set_prefix, '_', vowel_labels{I_vowel}, '.mat'], 'log_scores_train_LogReg_ss', 'log_scores_train_LogReg_ds', 'log_scores', 'Indices_comparisons');
    log_scores_cell{I_vowel} = log_scores;
    II_pairs_cell{I_vowel} = Indices_comparisons;
    log_scores_train_LogReg_ss_cell{I_vowel} = log_scores_train_LogReg_ss;
    log_scores_train_LogReg_ds_cell{I_vowel} = log_scores_train_LogReg_ds;
end

% sort data (assuming that all input has exactly the same set of comparisons, but not necessarily in the same order)
[II_pairs, II] = sortrows(II_pairs_cell{which_vowel(1)}, [1 2]);
num_comparisons = length(II_pairs);
num_iters = size(log_scores_cell{which_vowel(1)}, 2);
log_scores = NaN(num_comparisons, num_vowels, num_iters);
log_scores_train_LogReg_ss = cell(num_comparisons);
log_scores_train_LogReg_ds = cell(num_comparisons);

log_scores(:, 1, :) = log_scores_cell{which_vowel(1)}(II, :);
for I_vowel = which_vowel(2:end)
    for I_pairs = 1:num_comparisons
        I_row = II_pairs_cell{I_vowel}(:,1) == II_pairs(I_pairs, 1) & II_pairs_cell{I_vowel}(:,2) == II_pairs(I_pairs, 2);
        log_scores(I_pairs, I_vowel, :) = log_scores_cell{I_vowel}(I_row, :);
        log_scores_train_LogReg_ss{I_pairs}(:, I_vowel, :) =  log_scores_train_LogReg_ss_cell{I_vowel}{I_row};
        log_scores_train_LogReg_ds{I_pairs}(:, I_vowel, :) =  log_scores_train_LogReg_ds_cell{I_vowel}{I_row};
    end
end

% comparison-pair indices
II_ss = Indices_comparisons(:,1) == Indices_comparisons(:,2);
II_ds = ~II_ss;

% fuse
log_LR_fused_all_iters = NaN(num_comparisons, num_iters);
% cycle through speakers
fprintf('Fusing scores for %s data set.\n', data_set_prefix);
for I_comparison = 1:num_comparisons
    % cycle through number of UBM to Suspect model adaption steps
    for Iadapt = 1:num_iters
        % calculate calibration weights
        weights = train_llr_fusion_robust(log_scores_train_LogReg_ss{I_comparison}(:,:,Iadapt)', log_scores_train_LogReg_ds{I_comparison}(:,:,Iadapt)', 0.5, 0.001);
        % calibrate
        log_LR_fused_all(I_comparison, Iadapt) = lin_fusion(weights, log_scores(I_comparison, :, Iadapt)');
    end
end

% calculate Cllr
Cllr_fused_all = NaN(1, num_iters);
for Iadapt = 1:num_iters
    Cllr_fused_all(Iadapt) = cllr(log_LR_fused_all(II_ss, Iadapt), log_LR_fused_all(II_ds, Iadapt));
end
[Cllr_fused, best_num_iter] = min(Cllr_fused_all);
log_LR_fused = log_LR_fused_all(:, best_num_iter);
Cllr_min_fused = min_cllr(log_LR_fused(II_ss), log_LR_fused(II_ds));

% Tippett plot
plot_name = [data_set_prefix, ' fused'];
plot_tippett(exp(log_LR_fused(II_ss)), [], exp(log_LR_fused(II_ds)), [], plot_name, true, '-', true);
drawnow

% Output results to screen
fprintf('Cllr_fused: %0.3f\t Cllr_min_fused: %0.3f\t best_num_iter: %0.0f\n', Cllr_fused, Cllr_min_fused, best_num_iter');

% Save results
save_name = [data_set_prefix,'_fused'];
Indices_comparisons = II_pairs;
save(['.\resultsB\', save_name, '.mat'], 'log_LR_fused', 'Indices_comparisons', 'Cllr_fused', 'Cllr_min_fused', 'best_num_iter');
% saveas(gcf, ['.\plotsB\', save_name, '.fig']);

% clean up
rmpath('.\m_files', '.\m_files\cllr\', '.\m_files\fusion\');