function [log_scores_train_LogReg_ss, log_scores_train_LogReg_ds] = GMM_UBM_for_LogReg_train_limited_weights(data, Indices_Speakers, Indices_Sessions, num_gaussians, num_iter, avoid_non_psd, stat_options)

% speaker indices
speakerIDs = unique(Indices_Speakers);
numSpeakers = length(speakerIDs);

% session indices
session_1_indices = Indices_Sessions == 1;
session_2_indices = Indices_Sessions == 2;

% initiate variables
num_comparisons = (numSpeakers^2 + numSpeakers)/2;
log_scores = NaN(num_comparisons, 1);
indices_comparisons = NaN(num_comparisons, 2);

% cycle through speakers
I_speaker_pair = 0;
for Ispeaker_1 = 1:numSpeakers
    % speaker 1 training data (suspect)
    IIspeaker_1 = Indices_Speakers == speakerIDs(Ispeaker_1);
    II_train_1 = IIspeaker_1 & session_1_indices;
    training_data_1 = data(II_train_1, :);

    for Ispeaker_2 = Ispeaker_1:numSpeakers
        I_speaker_pair = I_speaker_pair + 1;

        % speaker 2 test data (offender)
        IIspeaker_2 = Indices_Speakers == speakerIDs(Ispeaker_2);
        II_not_test_speakers = ~(IIspeaker_1 | IIspeaker_2);
        II_test_2 = IIspeaker_2 & session_2_indices;
        test_data_2 = data(II_test_2, :);

        % training data (all other speakers)
        training_data = data(II_not_test_speakers, :);

        % build UBM
        lastwarn('');
        UBM = gmdistribution.fit(training_data, num_gaussians, 'CovType', 'diagonal', 'Regularize', avoid_non_psd, 'Options', stat_options);
        if ~isempty(lastwarn), beep, return, end % STOP here is there is a warning (failure to converge)
        UBM_likelihood = pdf(UBM, test_data_2);

        % adapt UBM to suspect model
        M_speaker = UBM;
        
        for Iadapt = 1:num_iter
            M_speaker = adaptUBM(M_speaker, training_data_1, avoid_non_psd);
        end
        M_speaker_likelihood = pdf(M_speaker, test_data_2);

        % get scores for individual test tokens
        scores_raw = M_speaker_likelihood ./ UBM_likelihood;
        % get mean log-scores, fixing any Inf values
        log_scores(I_speaker_pair) = mean(Inf2real(log(scores_raw)));

        % comparison indices
        indices_comparisons(I_speaker_pair, :) = [Ispeaker_1, Ispeaker_2];
    end
end

% comparison-pair indices
II_ss = indices_comparisons(:,1) == indices_comparisons(:,2);
II_ds = ~II_ss;

% output data for training logistic regression calibration / fusion
log_scores_train_LogReg_ss = log_scores(II_ss, :);
log_scores_train_LogReg_ds = log_scores(II_ds, :);
