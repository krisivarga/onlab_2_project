% run_MVKD2_likelihood_ratio_A.m 
%
% Aitken & Lucy (2004) 2-level multivariate kernel density model
% LR (score) output of MVKD2  are linearly calibrated to produce final LR output
%
% The procedure is symmetrical, but comparsions are restricted to the following:
% comparisons are session 1 (suspect model) with session 2 (offender data_kicsi) [same as session 2 (suspect model)with session 1 (offender data_kicsi)]
% for all same-speaker pairs, and all pairs of lower-numbered speaker (suspect) with higher numbered speaker (offender)
% e.g., Speaker 1 (suspect) vs Speaker 2 (offender), but not Speaker 2 (suspect) with Speaker 1 (offender)
%
% Two levels of cross-validation to ensure that data_kicsi from test speakers(s) is not used in training when calibrating and fusing
%
% data_kicsi are 3rd order DCT coefficients fitted to hertz-frequency original-time F2 trajectories

clear all
addpath('.\m_files', '.\m_files\cllr\', '.\m_files\fusion\');

% % data sets
% %                      1        2             3            4     
 vowel_labels = 'task_3_recordings';
% which_vowel = 4;
% 
% num_vowels = length(which_vowel);
% 
% % output to txt file
text_output_file = '.\results\MVKD2 Cllr results.txt';
fid = fopen(text_output_file, 'wt');
fprintf(fid, 'MVKD2 Cllr results\n');


%kiemel�s
%bg
load(['.\data_rand\session_3_train_data_formant3.mat'], 'Indices_Speakers', 'Indices_Sessions', 'data');    % speaker indices
bg_speakerIDs = unique(Indices_Speakers);
bg_speakerIDs_all = Indices_Speakers;
bg_numSpeakers = length(bg_speakerIDs);
bg_data = data;
bg_session_indices = Indices_Sessions;
%enrollment
load(['.\data_rand\session_1_test_data_formant3.mat'], 'Indices_Speakers', 'Indices_Sessions', 'data');    % speaker indices
enroll_speakerIDs = unique(Indices_Speakers);
enroll_speakerIDs_all = Indices_Speakers;
enroll_numSpeakers = length(enroll_speakerIDs);
enroll_data = data;
session_1_indices = Indices_Sessions;
%test
load(['.\data_rand\session_2_test_data_formant3.mat'], 'Indices_Speakers', 'Indices_Sessions', 'data');    % speaker indices
test_speakerIDs = unique(Indices_Speakers);
test_speakerIDs_all = Indices_Speakers;
test_numSpeakers = length(test_speakerIDs);
test_data = data;
session_2_indices = Indices_Sessions;

num_comparisons = (test_numSpeakers^2 + test_numSpeakers)/2;
scores_raw = NaN(num_comparisons, 1);
log_scores = NaN(num_comparisons, 1);
Indices_comparisons = NaN(num_comparisons, 2);
log_scores_train_LogReg_ss = cell(num_comparisons, 1);
log_scores_train_LogReg_ds = cell(num_comparisons, 1);
log_LR_cal = NaN(num_comparisons, 1);

fprintf('Calculating MVKD2 likelihood ratios for test set\n') %sz�vegen v�ltoztattam
fprintf('Started at %s\n', datestr(now))
I_speaker_pair = 0;
% %speaker 1 training data_kicsi (suspect)- marad�k 20 session 1 enroll  --- nem igy ,
% % mindig aktu�lis besz�l? lev�logat�sa
% IIspeaker_1 = enroll_speakerIDs_all; %megvan el?re
% II_train_1 = IIspeaker_1 & session_1_indices; % megvan el?re
% training_data_1 = enroll_data; %megvan el?re
% 
% 
% % speaker 2 test data (offender)- marad�k 20 session 2 test --- nem igy ,
% % mindig aktu�lis besz�l? lev�logat�sa
% IIspeaker_2 = test_speakerIDs_all; 
% %II_not_test_speakers = bg_speakerIDs;
% II_test_2 = IIspeaker_2 & session_2_indices;
% test_data_2 = test_data; %megvan el?re
            
% background data (all other speakers) ------------- bg/ubm?
background_data = bg_data; 
background_speaker_index = bg_speakerIDs_all; 
background_session_index = bg_session_indices;

for Ispeaker_1 = 1:enroll_numSpeakers
    %speaker 1 training data_kicsi (suspect)- marad�k 20 session 1 enroll
       
        IIspeaker_1 = enroll_speakerIDs_all == enroll_speakerIDs(Ispeaker_1); %megvan el?re
        II_train_1 = IIspeaker_1 & session_1_indices; % megvan el?re
        training_data_1 = data(II_train_1, :); %megvan el?re


        for Ispeaker_2 = 1:test_numSpeakers%--- �tl�ss�g oka
            fprintf('\nComparing speaker %0.0f against speaker %0.0f of %0.0f in data set all\n', Ispeaker_2, Ispeaker_1, test_numSpeakers)
            I_speaker_pair = I_speaker_pair + 1;
            
            IIspeaker_2 = test_speakerIDs_all == test_speakerIDs(Ispeaker_2); %megvan el?re, nem kell
            II_not_test_speakers = ~(IIspeaker_1 | IIspeaker_2); %megvan el?re, nem kell
            II_test_2 = IIspeaker_2 & session_2_indices; %nincs meg de legener�lhat� a megl�v? adatokb�l
            test_data_2 = data(II_test_2, :); %megvan el?re
            
            % MVKD2
            scores_raw(I_speaker_pair) = multivar_kernel_LR(training_data_1, test_data_2, background_data, background_speaker_index);
            log_scores(I_speaker_pair) = log(scores_raw(I_speaker_pair));
            
            % calibrate using cross-validated scores from background data-------------
            [log_scores_train_LogReg_ss{I_speaker_pair}, log_scores_train_LogReg_ds{I_speaker_pair}] = mvkd2_for_LogReg_train(background_data, background_speaker_index, background_session_index);
            % calculate calibration weights (handle cases of complete separation) ---------------
            weights = train_llr_fusion_robust(log_scores_train_LogReg_ss{I_speaker_pair}', log_scores_train_LogReg_ds{I_speaker_pair}', 0.5, 0.001);
            % calibrate
            log_LR_cal(I_speaker_pair) = lin_fusion(weights, log_scores(I_speaker_pair)');
            
            % comparison indices
            Indices_comparisons(I_speaker_pair, :) = [Ispeaker_1, Ispeaker_2];
            
            % estimated time to completion for this data_kicsi set
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
    Cllr_raw = cllr(log_scores(II_ss), log_scores(II_ds));
    Cllr_cal = cllr(log_LR_cal(II_ss), log_LR_cal(II_ds));
    Cllr_min_raw = min_cllr(log_scores(II_ss).', log_scores(II_ds).');
    Cllr_min_cal = min_cllr(log_LR_cal(II_ss).', log_LR_cal(II_ds).');
    
    % Tippett plot
    plot_name = ['MVKD2 ', vowel_labels];
    plot_tippett(exp(log_scores(II_ss)), [], exp(log_scores(II_ds)), [], [], true, '--', true);
    plot_tippett(exp(log_LR_cal(II_ss)), [], exp(log_LR_cal(II_ds)), [], plot_name, false, '-', false);
    drawnow
    
    % Output results to screen and text file
    fprintf('Cllr_raw: %0.3f\t Cllr_cal: %0.3f\t Cllr_min_raw: %0.3f\t Cllr_min_cal: %0.3f\n', Cllr_raw, Cllr_cal, Cllr_min_raw, Cllr_min_cal);
    fprintf(fid, '%s\nCllr_raw: %0.3f\t Cllr_cal: %0.3f\t Cllr_min_raw: %0.3f\t Cllr_min_cal: %0.3f\n', vowel_labels, Cllr_raw, Cllr_cal, Cllr_min_raw, Cllr_min_cal);
    
    % Save results
    save_name = ['MVKD2_', vowel_labels];
    save(['.\results\', save_name, '.mat'], 'log_scores_train_LogReg_ss', 'log_scores_train_LogReg_ds', 'log_scores', 'log_LR_cal', 'Indices_comparisons', 'Cllr_raw', 'Cllr_cal', 'Cllr_min_raw', 'Cllr_min_cal');
    saveas(gcf, ['.\plots\', save_name, '.fig']);


%clean up
rmpath('.\m_files', '.\m_files\cllr\', '.\m_files\fusion\');
fclose(fid);
% 
% 
% 
% % cycle through the data sets
% % for I_vowel = which_vowel
% %     % load data
% %     load(['.\data\', vowel_labels{I_vowel}, '.mat'], 'Indices_Speakers', 'Indices_Sessions', 'data');    % speaker indices
% %     speakerIDs = unique(Indices_Speakers);
% %     numSpeakers = length(speakerIDs);
% %     
% %     % session indices  %felesleges
% %     session_1_indices = Indices_Sessions == 1;
% %     session_2_indices = Indices_Sessions == 2;
%     
%     % initiate variables 
%     num_comparisons = (numSpeakers^2 + numSpeakers)/2;
%     scores_raw = NaN(num_comparisons, 1);
%     log_scores = NaN(num_comparisons, 1);
%     Indices_comparisons = NaN(num_comparisons, 2);
%     log_scores_train_LogReg_ss = cell(num_comparisons, 1);
%     log_scores_train_LogReg_ds = cell(num_comparisons, 1);
%     log_LR_cal = NaN(num_comparisons, 1);
%     
%     % cycle through speakers
%     fprintf('Calculating MVKD2 likelihood ratios for all\n')
%     fprintf('Started at %s\n', datestr(now))
%     I_speaker_pair = 0;
%     tic
%     for Ispeaker_1 = 1:numSpeakers
%         %speaker 1 training data_kicsi (suspect)- marad�k 20 session 1 enroll
%         %szerintem ezt m�r megcsin�lom el?re, teh�t felesleges l�p�s
%         IIspeaker_1 = Indices_Speakers == speakerIDs(Ispeaker_1); %megvan el?re
%         II_train_1 = IIspeaker_1 & session_1_indices; % megvan el?re
%         training_data_1 = data(II_train_1, :); %megvan el?re
% 
%         for Ispeaker_2 = Ispeaker_1:numSpeakers
%             fprintf('\nComparing speaker %0.0f against speaker %0.0f of %0.0f in data set all\n', Ispeaker_2, Ispeaker_1, numSpeakers)
%             I_speaker_pair = I_speaker_pair + 1;
%             
%             % speaker 2 test data (offender)- marad�k 20 session 2 test
%             IIspeaker_2 = Indices_Speakers == speakerIDs(Ispeaker_2); %megvan el?re, nem kell
%             II_not_test_speakers = ~(IIspeaker_1 | IIspeaker_2); %megvan el?re, nem kell
%             II_test_2 = IIspeaker_2 & session_2_indices; %nincs meg de legener�lhat� a megl�v? adatokb�l
%             test_data_2 = data(II_test_2, :); %megvan el?re
%             
%             % background data (all other speakers) ------------- bg/ubm?
%             background_data = data(II_not_test_speakers, :); %megvan el?re
%             background_speaker_index = Indices_Speakers(II_not_test_speakers); % megvan el?re
%             background_session_index = Indices_Sessions(II_not_test_speakers); % megvan el?re
%             
%             % MVKD2
%             scores_raw(I_speaker_pair) = multivar_kernel_LR(training_data_1, test_data_2, background_data, background_speaker_index);
%             log_scores(I_speaker_pair) = log(scores_raw(I_speaker_pair));
%             
%             % calibrate using cross-validated scores from background data-------------
%             [log_scores_train_LogReg_ss{I_speaker_pair}, log_scores_train_LogReg_ds{I_speaker_pair}] = mvkd2_for_LogReg_train(background_data, background_speaker_index, background_session_index);
%             % calculate calibration weights (handle cases of complete separation) ---------------
%             weights = train_llr_fusion_robust(log_scores_train_LogReg_ss{I_speaker_pair}', log_scores_train_LogReg_ds{I_speaker_pair}', 0.5, 0.001);
%             % calibrate
%             log_LR_cal(I_speaker_pair) = lin_fusion(weights, log_scores(I_speaker_pair)');
%             
%             % comparison indices
%             Indices_comparisons(I_speaker_pair, :) = [Ispeaker_1, Ispeaker_2];
%             
%             % estimated time to completion for this data_kicsi set
%             num_speaker_pairs_left = (num_comparisons - I_speaker_pair);
%             elapsed_time = toc;
%             est_completion_time = now + (elapsed_time * num_speaker_pairs_left / I_speaker_pair /86400); % 86400 = 24*60*60 i.e. number of seconds in a day
%             fprintf('\nEstimated completion time: %s\n', datestr(est_completion_time))
%         end
%     end
%     fprintf('Finished at %s\n', datestr(now))
%     
%     % comparison-pair indices
%     II_ss = Indices_comparisons(:,1) == Indices_comparisons(:,2);
%     II_ds = ~II_ss;
%     
%     % calculate Cllr
%     Cllr_raw = cllr(log_scores(II_ss), log_scores(II_ds));
%     Cllr_cal = cllr(log_LR_cal(II_ss), log_LR_cal(II_ds));
%     Cllr_min_raw = min_cllr(log_scores(II_ss).', log_scores(II_ds).');
%     Cllr_min_cal = min_cllr(log_LR_cal(II_ss).', log_LR_cal(II_ds).');
%     
%     % Tippett plot
%     plot_name = ['MVKD2 ', vowel_labels{I_vowel}];
%     plot_tippett(exp(log_scores(II_ss)), [], exp(log_scores(II_ds)), [], [], true, '--', true);
%     plot_tippett(exp(log_LR_cal(II_ss)), [], exp(log_LR_cal(II_ds)), [], plot_name, false, '-', false);
%     drawnow
%     
%     % Output results to screen and text file
%     fprintf('Cllr_raw: %0.3f\t Cllr_cal: %0.3f\t Cllr_min_raw: %0.3f\t Cllr_min_cal: %0.3f\n', Cllr_raw, Cllr_cal, Cllr_min_raw, Cllr_min_cal);
%     fprintf(fid, '%s\nCllr_raw: %0.3f\t Cllr_cal: %0.3f\t Cllr_min_raw: %0.3f\t Cllr_min_cal: %0.3f\n', vowel_labels{I_vowel}, Cllr_raw, Cllr_cal, Cllr_min_raw, Cllr_min_cal);
%     
%     % Save results
%     save_name = ['MVKD2_', vowel_labels{I_vowel}];
%     save(['.\results\', save_name, '.mat'], 'log_scores_train_LogReg_ss', 'log_scores_train_LogReg_ds', 'log_scores', 'log_LR_cal', 'Indices_comparisons', 'Cllr_raw', 'Cllr_cal', 'Cllr_min_raw', 'Cllr_min_cal');
%     saveas(gcf, ['.\plots\', save_name, '.fig']);
% 
% 
% % end
% % clean up
% rmpath('.\m_files', '.\m_files\cllr\', '.\m_files\fusion\');
% fclose(fid);