% run_GMM_likelihood_ratio_pooled.m 
%
% supect model means, covars, and weights are adapted from UBM using a variable number of iterations (best solution chosen)
% an LR is calculated for every token in the test data, these are converted to log, 
% and the mean of the log values used as the score for the comparison
% scores are then linearly calibrated to produce final LR output
%
% The procedure is asymmetrical, but comparsions are restricted to the following:
% comparisons are session 1 (suspect model) with session 2 (offender data)
% for all same-speaker pairs, and all pairs of lower-numbered speaker (suspect) with higher numbered speaker (offender)
% e.g., Speaker 1 (suspect) vs Speaker 2 (offender), but not Speaker 2 (suspect) with Speaker 1 (offender)
%
% Two levels of cross-validation to ensure that data from test speakers(s) is not used in training when calibrating and fusing
%
% Data are 3rd order DCT coefficients fitted to hertz-frequency original-time F2 trajectories

clear all
addpath('.\m_files', '.\m_files\cllr\', '.\m_files\fusion\');

% num Gaussians
num_gaussians_loop = [8, 9, 10, 11, 12, 13, 14, 15, 16];
label_loop = {'GMM08', 'GMM09', 'GMM10', 'GMM11', 'GMM12', 'GMM13', 'GMM14', 'GMM15', 'GMM16' };

% num_gaussians_loop = [59];
% label_loop = {'GMM59' };

output_dir = '.\results\';


% data sets
%                1    2    3    4    5    6    
vowel_labels = {'aI' 'eI' 'oU' 'aU' 'OI' 'all'};
% which_vowel = 1:5;
which_vowel = 6;

num_vowels = length(which_vowel);

% maximum number if iterations form adaptin UBM to Suspect model
max_adaption_iter = 15; %could hard code a single adaption_iter value for the exact method

% options for EM algorithm
stat_options = statset('MaxIter', 500);



for I_GM = 1:length(num_gaussians_loop)
    num_gaussians = num_gaussians_loop(I_GM);
    label = label_loop{I_GM};


%     % output to txt file
%     if num_gaussians == 15 %INCLUDES FIXES TO BE ABLE TO START IN MIDDLE
%         text_output_file = [output_dir, label, ' Cllr results.txt'];
%         fid = fopen(text_output_file, 'at');
%         which_vowel = [4:5];
%         num_vowels = length(which_vowel);
%     else
%         which_vowel = 1:5;
%         num_vowels = length(which_vowel);
%         % text_output_file = ['.\results\', label, ' Cllr results.txt'];
%         text_output_file = [output_dir, label, ' Cllr results.txt'];
%         fid = fopen(text_output_file, 'wt');
%     end    
    text_output_file = [output_dir, label, ' Cllr results.txt'];
    fid = fopen(text_output_file, 'wt');
    
    fprintf(fid, [label, ' Cllr results\n']);

    % cycle through the data sets
    for I_vowel = which_vowel
        % load data
        load(['.\data\', vowel_labels{I_vowel}, '.mat'], 'Indices_Speakers', 'Indices_Sessions', 'data');

        % speaker indices
        speakerIDs = unique(Indices_Speakers);
        numSpeakers = length(speakerIDs);

        % session indices
        session_1_indices = Indices_Sessions == 1;
        session_2_indices = Indices_Sessions == 2;

        % data
        num_vars = size(data, 2);

        % avoid non-positive-semi-definite covariance matrices
        avoid_non_psd = 1e-12;
        avoid_non_psd_diag = eye(num_vars) * avoid_non_psd;

        % initiate variables
        num_comparisons = (numSpeakers^2 + numSpeakers)/2;
        log_scores = NaN(num_comparisons, max_adaption_iter);
        Indices_comparisons = NaN(num_comparisons, 2);
        log_scores_train_LogReg_ss = cell(num_comparisons, 1);
        log_scores_train_LogReg_ds = cell(num_comparisons, 1);
        log_LR_cal_all = NaN(num_comparisons, max_adaption_iter);

        % cycle through speakers
        fprintf('Calculating %s likelihood ratios for %s\n', label, vowel_labels{I_vowel})
        fprintf('Started at %s\n', datestr(now))
        I_speaker_pair = 0;
        tic
        for Ispeaker_1 = 1:numSpeakers
            % speaker 1 training data (suspect)
            IIspeaker_1 = Indices_Speakers == speakerIDs(Ispeaker_1);
            II_train_1 = IIspeaker_1 & session_1_indices;
            training_data_1 = data(II_train_1, :);

            for Ispeaker_2 = Ispeaker_1:numSpeakers
                fprintf('\nComparing speaker %0.0f against speaker %0.0f of %0.0f in data set %s\n', Ispeaker_2, Ispeaker_1, numSpeakers, vowel_labels{I_vowel})
                I_speaker_pair = I_speaker_pair + 1;

                % speaker 2 test data (offender)
                IIspeaker_2 = Indices_Speakers == speakerIDs(Ispeaker_2);
                II_not_test_speakers = ~(IIspeaker_1 | IIspeaker_2);
                II_test_2 = IIspeaker_2 & session_2_indices;
                test_data_2 = data(II_test_2, :);

                % training data (all other speakers)
                training_data = data(II_not_test_speakers, :);
                training_speaker_index = Indices_Speakers(II_not_test_speakers);
                training_session_index = Indices_Sessions(II_not_test_speakers);

                % build UBM
                lastwarn('');
                UBM = gmdistribution.fit(training_data, num_gaussians, 'CovType', 'diagonal', 'Regularize', avoid_non_psd, 'Options', stat_options);
                if ~isempty(lastwarn), beep, return, end % STOP here is there is a warning (failure to converge)
                UBM_likelihood = pdf(UBM, test_data_2);

                % adapt UBM to suspect model
                M_speaker = UBM;

                % get log_scores to use for calibration
                [log_scores_train_LogReg_ss{I_speaker_pair}, log_scores_train_LogReg_ds{I_speaker_pair}] = GMM_UBM_for_LogReg_train(training_data, training_speaker_index, training_session_index, num_gaussians, max_adaption_iter, avoid_non_psd, stat_options);

                for Iadapt = 1:max_adaption_iter
                    M_speaker = adaptUBM(M_speaker, training_data_1, avoid_non_psd);
                    M_speaker_likelihood = pdf(M_speaker, test_data_2);

                    % get scores for individual test tokens
                    scores_raw = M_speaker_likelihood ./ UBM_likelihood;
                    % get mean log-scores, fixing any Inf values
                    log_scores(I_speaker_pair, Iadapt) = mean(Inf2real(log(scores_raw)));

                    % calculate calibration weights (handle cases of complete separation)
                    weights = train_llr_fusion_robust(log_scores_train_LogReg_ss{I_speaker_pair}(:, Iadapt)', log_scores_train_LogReg_ds{I_speaker_pair}(:, Iadapt)', 0.5, 0.001);
                    % calibrate
                    log_LR_cal_all(I_speaker_pair, Iadapt) = lin_fusion(weights, log_scores(I_speaker_pair, Iadapt)');
                end

                % comparison indices
                Indices_comparisons(I_speaker_pair, :) = [Ispeaker_1, Ispeaker_2];

                % estimated time to completion for this data set
                num_speaker_pairs_left = (num_comparisons - I_speaker_pair);
                elapsed_time = toc;
                est_completion_time = now + (elapsed_time * num_speaker_pairs_left / I_speaker_pair /86400); % 86400 = 24*60*60 i.e. number of seconds in a day
                fprintf('\nEstimated completion time: %s\n', datestr(est_completion_time))
            end
        end
        fprintf('Finished at %s\n', datestr(now))

        % comparison-pair indices
        II_ss = Indices_comparisons(:,1) == Indices_comparisons(:,2);
        II_ds = ~II_ss;

        % calculate Cllr
        Cllr_raw_all = NaN(max_adaption_iter, 1);
        Cllr_cal_all = NaN(max_adaption_iter, 1);
        Cllr_min_raw_all = NaN(max_adaption_iter, 1);
        Cllr_min_cal_all = NaN(max_adaption_iter, 1);
        for Iadapt = 1:max_adaption_iter
            Cllr_raw_all(Iadapt) = cllr(log_scores(II_ss, Iadapt), log_scores(II_ds, Iadapt));
            Cllr_cal_all(Iadapt) = cllr(log_LR_cal_all(II_ss, Iadapt), log_LR_cal_all(II_ds, Iadapt));
            Cllr_min_raw_all(Iadapt) = min_cllr(log_scores(II_ss, Iadapt), log_scores(II_ds, Iadapt));
            Cllr_min_cal_all(Iadapt) = min_cllr(log_LR_cal_all(II_ss, Iadapt), log_LR_cal_all(II_ds, Iadapt));
        end

        % best Cllr_cal
        [Cllr_cal, best_num_iter] = min(Cllr_cal_all);
        Cllr_raw = Cllr_raw_all(best_num_iter);
        Cllr_min_raw = Cllr_min_raw_all(best_num_iter);
        Cllr_min_cal = Cllr_min_cal_all(best_num_iter);
        log_LR_cal = log_LR_cal_all(:, best_num_iter);

    %     % Tippett plot
    %     plot_name = [label, ' ', vowel_labels{I_vowel}];
    %     plot_tippett(exp(log_LR_cal(II_ss)), [], exp(log_LR_cal(II_ds)), [], plot_name, true, '-', true);
    %     drawnow

        % Output results to screen and text file
        fprintf('Cllr_raw: %0.3f\t Cllr_cal: %0.3f\t Cllr_min_raw: %0.3f\t Cllr_min_cal: %0.3f\t best_num_iter: %0.0f\n', Cllr_raw, Cllr_cal, Cllr_min_raw, Cllr_min_cal, best_num_iter);
        fprintf(fid, '%s\nCllr_raw: %0.3f\t Cllr_cal: %0.3f\t Cllr_min_raw: %0.3f\t Cllr_min_cal: %0.3f\t best_num_iter: %0.0f\n', vowel_labels{I_vowel}, Cllr_raw, Cllr_cal, Cllr_min_raw, Cllr_min_cal, best_num_iter);

        % Save results
        save_name = [label, '_', vowel_labels{I_vowel}];
        save([output_dir, save_name, '.mat'], 'log_scores_train_LogReg_ss', 'log_scores_train_LogReg_ds', 'log_scores', 'log_LR_cal_all', 'Cllr_raw_all', 'Cllr_cal_all', 'Cllr_min_raw_all', 'Cllr_min_cal_all', 'Indices_comparisons', 'log_LR_cal', 'Cllr_raw', 'Cllr_cal', 'Cllr_min_raw', 'Cllr_min_cal', 'best_num_iter');
%         saveas(gcf, ['.\plots\', save_name, '.fig']);
    end

    
    
end

% clean up
rmpath('.\m_files', '.\m_files\cllr\', '.\m_files\fusion\');
fclose(fid);