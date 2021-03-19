% find_best.m
%
% find best combination of num GMMs and num iterations

clear all

vowel_labels = {'aI' 'eI' 'oU' 'aU' 'OI' 'all'};
which_vowel = 1;
num_vowels = length(which_vowel);

dir_name = '.\results\';

% hard coded
num_files = 9;
num_mix = 9;
num_iter = 15;

Cllr_cal = cell(1, num_vowels);
min_Cllr_cal = NaN(num_vowels, 1);
min_mix = NaN(num_vowels, 1);
min_iter = NaN(num_vowels, 1);
for I_vowel = 1:num_vowels
    file_search = [dir_name, 'GMM*_', vowel_labels{which_vowel(I_vowel)}, '.mat'];
    tdir = dir(file_search);
    % num_files = length(tdir);
    [file_names{1:num_files}] = deal(tdir.name);
    % num_mix = num_files;
    
    Cllr_cal{I_vowel} = NaN(num_mix, num_iter);
    for I_mix = 1:num_mix
        mix(I_mix) = str2num(file_names{I_mix}(4:5));
        load([dir_name, file_names{I_mix}], 'Cllr_cal_all');
        Cllr_cal{I_vowel}(I_mix, :) = Cllr_cal_all;
    end
    figure(I_vowel);
    surf(Cllr_cal{I_vowel});
    title(vowel_labels{which_vowel(I_vowel)});
    ylabel('num mixtures');
    xlabel('num iterations');
    set(gca, 'YTick', [1:num_mix], 'YTickLabel', mix, 'XTick', [1:num_iter]);
    xlim([1 num_iter]);
    ylim([1 num_mix]);
    zlim([0 0.8]) % hard coded
    zlabel('Cllr calibrated');
    [min_Cllr_cal_temp, min_mix_temp] = min(Cllr_cal{I_vowel});
    [min_Cllr_cal(I_vowel), min_iter(I_vowel)] = min(min_Cllr_cal_temp);
    min_mix(I_vowel) = min_mix_temp(min_iter(I_vowel));
    hold on
    stem3(min_iter(I_vowel), min_mix(I_vowel), min_Cllr_cal(I_vowel), '.r');
    hold off
end

print_matrix = [min_Cllr_cal, mix(min_mix)', min_iter]';
fprintf('%0.3f\t%0.0f\t%0.0f\n', print_matrix);