% run_GMM_likelihood_ratio_B_limited.m 
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


output_dir = '.\results\';

% data sets
%                1      2     3     4     5    
vowel_labels = {'aI'   'eI'  'oU'  'aU'  'OI'};
num_gauss =     [14    10    11    10    14];
num_iter =      [11     7     6    10     8];

% which_vowel = 1:5;
which_vowel = 2:5;

num_vowels = length(which_vowel);


% options for EM algorithm
stat_options = statset('MaxIter', 500);



% cycle through the data sets
for I_vowel = which_vowel
    % load data
    load(['.\data\', vowel_labels{I_vowel}, '.mat'], 'Indices_Speakers', 'Indices_Sessions', 'data');

    % speaker indices
    speakerIDs = unique(Indices_Speakers);
    numSpeakers = length(speakerIDs);

    % session indices REVERSED
    session_1_indices = Indices_Sessions == 2;
    session_2_indices = Indices_Sessions == 1;

    % data
    num_vars = size(data, 2);

    % avoid non-positive-semi-definite covariance matrices
    avoid_non_psd = 1e-12;
    avoid_non_psd_diag = eye(num_vars) * avoid_non_psd;

    % initiate variables
    num_comparisons = (numSpeakers^2 + numSpeakers)/2;
    log_scores = NaN(num_comparisons, 1);
    Indices_comparisons = NaN(num_comparisons, 2);
    log_scores_train_LogReg_ss = cell(num_comparisons, 1);
    log_scores_train_LogReg_ds = cell(num_comparisons, 1);
    log_LR_cal_all = NaN(num_comparisons, 1);

    % cycle through speakers
    fprintf('Calculating likelihood ratios for %s\n', vowel_labels{I_vowel})
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
            UBM = gmdistribution.fit(training_data, num_gauss(I_vowel), 'CovType', 'diagonal', 'Regularize', avoid_non_psd, 'Options', stat_options);
            if ~isempty(lastwarn), beep, return, end % STOP here is there is a warning (failure to converge)
            UBM_likelihood = pdf(UBM, test_data_2);

            % adapt UBM to suspect model
            M_speaker = UBM;
            for Iadapt = 1:num_iter(I_vowel)
                M_speaker = adaptUBM(M_speaker, training_data_1, avoid_non_psd);
                M_speaker_likelihood = pdf(M_speaker, test_data_2);
            end

            % get scores for individual test tokens
            scores_raw = M_speaker_likelihood ./ UBM_likelihood;
            
            % get mean log-scores, fixing any Inf values
            log_scores(I_speaker_pair) = mean(Inf2real(log(scores_raw)));

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

    % Save results
    save_name = ['b_GMM', num2str(num_gauss(I_vowel), '%02.0f'), '_', num2str(num_iter(I_vowel), '%02.0f'), '_', vowel_labels{I_vowel}];
    save([output_dir, save_name, '.mat'], 'log_scores', 'Indices_comparisons');
end

% clean up
rmpath('.\m_files', '.\m_files\cllr\', '.\m_files\fusion\');
