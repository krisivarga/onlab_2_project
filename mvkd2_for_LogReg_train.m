function [log_scores_train_LogReg_ss, log_scores_train_LogReg_ds] = mvkd2_for_LogReg_train(data, Indices_Speakers, Indices_Sessions)

% speaker indices
speakerIDs = unique(Indices_Speakers);
numSpeakers = length(speakerIDs);

% session indices
session_1_indices = Indices_Sessions == 1;
session_2_indices = Indices_Sessions == 2;

% initiate variables
num_comparisons = (numSpeakers^2 + numSpeakers)/2;
scores = NaN(num_comparisons, 1);
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

        % background data (all other speakers)
        background_data = data(II_not_test_speakers, :);
        background_speaker_index = Indices_Speakers(II_not_test_speakers);

        % MVKD2
        scores(I_speaker_pair) = multivar_kernel_LR(training_data_1, test_data_2, background_data, background_speaker_index);

        % comparison indices
        indices_comparisons(I_speaker_pair, :) = [Ispeaker_1, Ispeaker_2];
    end
end

% log scores
log_scores = log(scores);

% comparison-pair indices
II_ss = indices_comparisons(:,1) == indices_comparisons(:,2);
II_ds = ~II_ss;

% output data for training logistic regression calibration / fusion
log_scores_train_LogReg_ss = log_scores(II_ss);
log_scores_train_LogReg_ds = log_scores(II_ds);
