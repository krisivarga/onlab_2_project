function plot_tippett(likelihood_ratio_same_speaker, IIsame_speaker_compared, likelihood_ratio_different_speaker, IIdifferent_speaker_compared, figure_title, new_fig, line_type, zero_line, dont_sort)
% plot_tippett(likelihood_ratio_same_speaker, IIsame_speaker_compared, likelihood_ratio_different_speaker, IIdifferent_speaker_compared, figure_title, new_fig, line_type, zero_line)
%
% DONT_SORT option for drawing credible intervals
%
% by Geoffrey Stewart Morrison  http://geoff-morrison.net
% Version date: 2008-11-13
% sucessfully tested using Matlab R2007b, R2008a, R2008b running under Windows XP x64
%
% Function which draws Tippett plots
%   Red line: proportion of different-speaker comparisons with log10 likelihood ratio equal to or less than value indicated on x-axis
%   Blue line: proportion of same-speaker comparisons with log10 likelihood ratio equal to or greater than value indicated on x-axis
%
% REQUIRED INPUTS:
%   likelihood_ratio_same_speaker           vector of likelihood ratios from same-speaker comparisons
%   IIsame_speaker_compared                 not used in this version but may be of use in future versions*
%   likelihood_ratio_different_speaker      vector of likelihood ratios from same-different comparisons
%   IIdifferent_speaker_compared            not used in this version but may be of use in future versions*
%                                           *call function usind empty arrays: plot_tippett(likelihood_ratio_same_speaker, [], likelihood_ratio_different_speaker, [], figure_title);
% OPTIONAL INPUTS:
%   figure_title                            title for Tippet plot (optional)
%   new_fig                                 logical true to plot on a new figure [defualt] , false to plot on an existing figure
%   line_type                               '-' solid [default], '--' dashed, ':' dotted, '-.' dashed-dotted
%   zero_line                               logical true to daw vertical line at log10LR = 0 [defualt] , false to not plot line

if nargin == 0      % make a plot using sample data
    LR_ss = [.8 1 5 3 9 10 25 6 20 18 .9 .7 11];
    LR_ds = [.002 .01 .3 .5 .9 1.2 .6 .05 .006 1.3 .4 .2 .03 1.1 2 .0005];
    plot_tippett(LR_ss, [], LR_ds, []);
    return
end

if nargin < 5, figure_title = 'Tippett Plot'; end   % default figure title
if nargin < 6, new_fig = true; end   % default plot figure in new window
if nargin < 7, line_type = '-'; end   % default solid line
if nargin < 8, zero_line = true; end   % default vertical line at zero
if nargin < 9, dont_sort = false; end   % default sort input

% put LR data is in form of column vectors
% sort LR data
% calculate log10 of LR data
% calculate cumulative proportions
likelihood_ratio_different_speaker = likelihood_ratio_different_speaker(:);
if dont_sort
    LR_sorted_different_speaker = likelihood_ratio_different_speaker;
else
%     [LR_sorted_different_speaker, II_sorted_different_speaker] = sort(likelihood_ratio_different_speaker);
    LR_sorted_different_speaker = sort(likelihood_ratio_different_speaker);
end
log_LR_sorted_different_speaker = log10(LR_sorted_different_speaker);
num_different_speaker_compared = length(likelihood_ratio_different_speaker);
cumulative_proportion_different_speaker = (num_different_speaker_compared:-1:1)/num_different_speaker_compared;

likelihood_ratio_same_speaker = likelihood_ratio_same_speaker(:);
if dont_sort
    LR_sorted_same_speaker = likelihood_ratio_same_speaker;
else
%     [LR_sorted_same_speaker, II_sorted_same_speaker] = sort(likelihood_ratio_same_speaker);
    LR_sorted_same_speaker = sort(likelihood_ratio_same_speaker);
end
log_LR_sorted_same_speaker = log10(LR_sorted_same_speaker);
num_same_speaker_compared = length(likelihood_ratio_same_speaker);
cumulative_proportion_same_speaker = (1:num_same_speaker_compared)/num_same_speaker_compared;

% plot cumulative proportions against log10 LR values
if new_fig, figure; else hold on, end
if zero_line
    plot([0 0], [0 1], 'Color', 'g', 'LineStyle', '-');
    hold on
end
plot(log_LR_sorted_different_speaker, cumulative_proportion_different_speaker, 'Color', 'r', 'LineStyle', line_type);
hold on
plot(log_LR_sorted_same_speaker, cumulative_proportion_same_speaker, 'Color', 'b', 'LineStyle', line_type);
hold off
title(figure_title)
xlabel('Log10 Likelihood Ratio')
ylabel('Cumulative Proportion')

return

