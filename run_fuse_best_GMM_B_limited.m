% run_fuse_best_GMM_B_limited.m

clear all
addpath('.\m_files', '.\m_files\cllr\', '.\m_files\fusion\');

complete_separation = true;

% data sets
%                1      2     3     4     5    
vowel_labels = {'aI'   'eI'  'oU'  'aU'  'OI'};
mix =           [14    10    11    10    14];
iter =          [11     7     6    10     8];

num_vowels = 5;

% load data
log_scores_cell = cell(1, num_vowels);
II_pairs_cell = cell(1, num_vowels);
log_scores_train_LogReg_ss_cell = cell(1, num_vowels);
log_scores_train_LogReg_ds_cell = cell(1, num_vowels);
for I_vowel = 1:num_vowels
    load(['.\results\GMM', num2str(mix(I_vowel), '%02.0f'), '_', vowel_labels{I_vowel}, '.mat'], 'log_scores_train_LogReg_ss', 'log_scores_train_LogReg_ds', 'Indices_comparisons');
    load(['.\results\b_GMM', num2str(mix(I_vowel), '%02.0f'), '_', num2str(iter(I_vowel), '%02.0f'), '_', vowel_labels{I_vowel}, '.mat'], 'log_scores', 'Indices_comparisons');
    log_scores_cell{I_vowel} = log_scores; 
    II_pairs_cell{I_vowel} = Indices_comparisons; % assume order of comparisons is same for all phonemes and both LR sets
    log_scores_train_LogReg_ss_cell{I_vowel} = cellfun(@(x) x(:, iter(I_vowel)), log_scores_train_LogReg_ss, 'UniformOutput', false);
    log_scores_train_LogReg_ds_cell{I_vowel} = cellfun(@(x) x(:, iter(I_vowel)), log_scores_train_LogReg_ds, 'UniformOutput', false);
end
log_scores = cell2mat(log_scores_cell);

% comparison-pair indices
II_ss = Indices_comparisons(:,1) == Indices_comparisons(:,2);
II_ds = ~II_ss;

num_comparisons = length(II_ss);

% fuse
log_LR_fused = NaN(num_comparisons, 1);
% cycle through speakers
for I_comparison = 1:num_comparisons
    % extract data
    log_scores_train_LogReg_ss_temp = cell(1, num_vowels);
    log_scores_train_LogReg_ds_temp = cell(1, num_vowels);
    for Ivowel = 1:num_vowels
        log_scores_train_LogReg_ss_temp(Ivowel) = log_scores_train_LogReg_ss_cell{Ivowel}(I_comparison);
        log_scores_train_LogReg_ds_temp(Ivowel) = log_scores_train_LogReg_ds_cell{Ivowel}(I_comparison);
    end
    log_scores_train_LogReg_ss_temp = cell2mat(log_scores_train_LogReg_ss_temp);
    log_scores_train_LogReg_ds_temp = cell2mat(log_scores_train_LogReg_ds_temp);
    % calculate calibration weights
    weights = train_llr_fusion_robust(log_scores_train_LogReg_ss_temp', log_scores_train_LogReg_ds_temp', 0.5, 0.001);
    % calibrate
    log_LR_fused(I_comparison) = lin_fusion(weights, log_scores(I_comparison, :)');
end

% calculate Cllr
Cllr_fused = cllr(log_LR_fused(II_ss), log_LR_fused(II_ds));
Cllr_min_fused = min_cllr(log_LR_fused(II_ss), log_LR_fused(II_ds));

% Tippett plot
plot_name = 'GMM-UBM fused';
plot_tippett(exp(log_LR_fused(II_ss)), [], exp(log_LR_fused(II_ds)), [], plot_name, true, '-', true);
drawnow

% Output results to screen
fprintf('Cllr_fused: %0.3f\t Cllr_min_fused: %0.3f\n', Cllr_fused, Cllr_min_fused);

% Save results
save_name = 'b_GMM_fused';
save(['.\results\', save_name, '.mat'], 'log_LR_fused', 'Indices_comparisons', 'Cllr_fused', 'Cllr_min_fused');
% saveas(gcf, ['.\plots\', save_name, '.fig']);

% clean up
rmpath('.\m_files', '.\m_files\cllr\', '.\m_files\fusion\');