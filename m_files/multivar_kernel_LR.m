function likelihood_ratio = multivar_kernel_LR(suspect_data, offender_data, background_data, background_index)
%
% This implementation (c) Geoffrey Stewart Morrison 2007
% Version of 25 March 2008
% change since version of 24 November 2007: Output cannot be less than smallest floating-point number.
%
% Implementation of the multivariate kernel-density likelihood ratio software described in: 
% Aitken, C. G. G., & Lucy, D. (2004). Evaluation of trace evidence in the form of multivariate data. Appl. Stat. 53, Part 1, pp. 109-122.
% 
% Nomencalture of variable names and comments is designed with a forensic speaker recognition task in mind. 
% This implementation does not require an equal number of measurements from each speaker in the background sample.
%
% REQUIRED TOOLBOX:
%       - Statistics toolbox
%
% VARIABLES:
% [variable name in code]  [var name in A&L(2004)]      [description]
%
% background_data           x       array of measurments from background sample - each row represents a case/token - each colum represents a different variable
% background_index          i       indices of speakers (sources) in background population - correspond to columns of background_data - may be a numerical vector, a string array, or a cell array of strings
% speaker_names                     unique values from background_index
% speaker_indices                   indices of speaker_names
% Ispeaker                          for-loop index value for stepping through speakers
% background_data_per_speaker       cell array of background_data divided on a per speaker basis
% 
% num_speakers              m       number of speakers (sources) in background data
% num_measures              n       number of measures (cases, tokens) per speaker of phoneme etc. of interest (replications per source)in background data -* an equal number of measures per speaker is not assumed and this variable is not used in the code
% num_observations          N       total number of observations (cases, tokens)
% num_variables             p       number of variables measured
%
% background_means          x-bar-i array of within-speaker (within-source) variable means - each row represents a speaker - each colum represents a variable
% background_grand_mean     x-bar   vector of variable means across all speakers (sources) -* this varibale not used in the code
% within_covar              U       within-speaker (within-source) covariance matrix (assumed to be same for each speaker)
% within_covar_raw                  intermediate calculation for within_covar
% background_within_covar   U/n     within-speaker covariance matrix divided by num measures per speaker - see notes in code on how this is calculated
% background_within_covar_raw       intermediate calculation for background_within_covar
% background_between_covar  C       between-speaker (between-source) covariance matrix (assuming normality, but will also be used in the kernel density function)
% background_between_covar_raw      intermediate calculation for background_between_covar
%
% suspect_data              y2      array of measurments from suspect (recovered) sample - each row represents a case/token - each colum represents a different variable
% suspect_mean              y2-bar  vector of variable means for suspect
% suspect_num_measures      n2      number of measures (cases, tokens) of phoneme etc. of interest (replications per source)in suspect data
% suspect_covar             D2      within-speaker covariance matrix from background population divided by suspect_num_measures
% suspect_covar_inv                 inverse of suspect_covar
%
% offender_data             y1      array of measurments from offender (scene, control) sample - each row represents a case/token - each colum represents a different variable
% offender_mean             y1-bar  vector of variable means for offender
% offender_num_measures     n1      number of measures (cases, tokens) of phoneme etc. of interest (replications per source)in offender data
% offender_covar            D1      within-speaker covariance matrix from background population divided by offender_num_measures
% offender_covar_inv                inverse of offender_covar
%
% smoothing_parameter       h       smoothing parameter for kernel density calculation
% smooth_power                      intermediate calculation for smoothing_parameter
% kernel                            smoothing_parameter^2 * background_between_covar
% inv_kernel                        inverse of kernel
%
% suspect_offender_mean_difference  offender_mean minus suspect_mean
% suspect_offender_mean_typicality  y* in A&L(2004)
% kernel_density_at_typicality      kernel density funtion evaluated at suspect_offender_mean_typicality
% typicality                        distance from suspect_offender_mean_typicality of a background speaker
%
% dist_backspeakers_to_suspect      average distance from background speakers to suspect
% dist_to_suspect                   intermediate stage in calculation of dist_backspeakers_to_suspect
% dist_backspeakers_to_offender     average distance from background speakers to offender
% dist_to_offender                  intermediate stage in calculation of dist_backspeakers_to_offender
%
% numerator                         numerator of likelihood ratio
% denominator                       denominator of likelihood ratio
% likelihood_ratio                  likelihood ratio to return


%% TEST DATA

if nargin > 0 && nargin < 4
    error('Incorrect number of input arguments for function: likelihood_ratio = multivar_kernel_LR(suspect_data, offender_data, background_data, background_index)')
elseif nargin == 0
    disp('Running test data on function: likelihood_ratio = multivar_kernel_LR(suspect_data, offender_data, background_data, background_index)')
    background_index=[1 1 1 2 2 2 3 3 3 4 4 4];
    background_data=[10 21 30
                     11 20 31
                     12 22 32
                     12 22 33
                     13 23 32
                     14 24 34
                     15 24 34
                     14 25 35
                     16 26 36
                     16 26 36
                     17 27 37
                     18 28 38];
    suspect_data=[11 21 31
                  12 22 32
                  13 23 33];

    disp('offender_data=suspect_data+.1')
    offender_data=suspect_data+.1;
    LR = multivar_kernel_LR(suspect_data, offender_data, background_data, background_index)

    disp('offender_data=suspect_data+1')
    offender_data=suspect_data+1;
    LR = multivar_kernel_LR(suspect_data, offender_data, background_data, background_index)

    disp('offender_data=suspect_data+5')
    offender_data=suspect_data+5;
    LR = multivar_kernel_LR(suspect_data, offender_data, background_data, background_index)
    return
end


%% BACKGROUND 

% number of variables and number of observations
[num_observations,num_variables] = size(background_data);

% speaker indices and number of speakers
[speaker_indices,speaker_names] = grp2idx(background_index);
num_speakers = length(speaker_names);

% divide data on per-speaker basis - need to use a cell approach because an equal number of tokens per speaker is not assumed
background_data_per_speaker = cell(num_speakers,1);
for Ispeakers = 1:num_speakers
    background_data_per_speaker{Ispeakers} = background_data(speaker_indices==Ispeakers,:);
end

% within-speaker and across-speaker means
background_means = cell2mat(cellfun(@mean, background_data_per_speaker, 'UniformOutput', false) ); 

% within and between speaker covariance matrices
within_covar_raw = cellfun(@sse, background_data_per_speaker, 'UniformOutput', false);
within_covar = sum(reshape(cell2mat(within_covar_raw)',[num_variables,num_variables,num_speakers]),3)/(num_observations-num_speakers);

background_within_covar_raw = cellfun(@cov, background_data_per_speaker, 'UniformOutput', false);
background_within_covar = sum(reshape(cell2mat(background_within_covar_raw)',[num_variables,num_variables,num_speakers]),3)/num_observations;
    % EXPLANATION OF CALCULATIONS NOT ASSUMING EQUAL n
    % background_within_covar is second part of equation 2 in A&L(2004), i.e., Sw/(n*(N-m)) where n is the number of tokens per speaker
    % note that Sw = sum(sse(x)) = sum(cov(x) * (n-1)), and the total number of tokens (observations) N = n*m
    % therefore Sw/(n*(N-m)) = sum(cov(x) * (n-1)) / (n*(N-m)) = sum(cov(x)) * (n-1) / (n * m*(n-1)) = sum(cov(x)) / N
    % sum(cov(x))/N can be calculated without the assumption that there is an equal number of tokens per speaker

background_between_covar_raw = cov(background_means);
background_between_covar = background_between_covar_raw - background_within_covar;

% % IF AN EQUAL n PER SPEAKER HAD BEEN ASSUMED
% num_measures = size(background_data_per_speaker{1},1);
% background_within_covar_equal_n = within_covar/num_measures;
% background_between_covar_raw_equal_n = sse(background_means)/(num_speakers-1);
% background_between_covar_equal_n = background_between_covar_raw_equal_n - background_within_covar_equal_n;
% test = background_between_covar - background_between_covar_equal_n;

%% SUSPECT AND OFENDER

suspect_num_measures = size(suspect_data,1);
suspect_mean = mean(suspect_data,1)';
suspect_covar = within_covar/suspect_num_measures;
suspect_covar_inv = inv(suspect_covar);

offender_num_measures = size(offender_data,1);
offender_mean = mean(offender_data,1)';
offender_covar = within_covar/offender_num_measures;
offender_covar_inv = inv(offender_covar);

suspect_offender_mean_difference = offender_mean-suspect_mean;
suspect_offender_mean_typicality = (offender_covar_inv+suspect_covar_inv) \ (offender_covar\offender_mean + suspect_covar\suspect_mean);


%% LIKELIHOOD RATIO

smooth_power = 1/(num_variables+4);
smoothing_parameter = (4/(2*num_variables+1))^smooth_power * num_speakers^-smooth_power;
kernel = smoothing_parameter^2 * background_between_covar;
inv_kernel = inv(kernel);

kernel_density_at_typicality = 0;
dist_backspeakers_to_suspect = 0;
dist_backspeakers_to_offender = 0;
for Ispeaker = 1:num_speakers
    typicality = suspect_offender_mean_typicality - background_means(Ispeaker,:)';
    kernel_density_at_typicality = kernel_density_at_typicality + exp(-.5 * typicality' * inv( inv(offender_covar_inv + suspect_covar_inv) + kernel ) * typicality);

    dist_to_suspect = suspect_mean - background_means(Ispeaker,:)';
    dist_backspeakers_to_suspect = dist_backspeakers_to_suspect + exp(-.5 * dist_to_suspect' * inv(suspect_covar + kernel) * dist_to_suspect);

    dist_to_offender = offender_mean - background_means(Ispeaker,:)';
    dist_backspeakers_to_offender = dist_backspeakers_to_offender + exp(-.5 * dist_to_offender' * inv(offender_covar + kernel) * dist_to_offender);
end

    % have fixed problems resulting from negative determinants, and hence complex numbers when square rooted, by using absolute values of determinants - A&L's solution in R software was to use prod(eigen(A)$values)
numerator = (2*pi)^-num_variables * det(offender_covar)^-.5 * det(suspect_covar)^-.5 * abs(det(background_between_covar))^-.5 ...
            * (num_speakers * smoothing_parameter^num_variables)^-1 * abs(det(offender_covar_inv + suspect_covar_inv + inv_kernel))^-.5 ...
            * exp(-.5 * suspect_offender_mean_difference' * inv(offender_covar + suspect_covar) * suspect_offender_mean_difference) ...
            * kernel_density_at_typicality ;
            
denominator = (2*pi)^-num_variables * abs(det(background_between_covar))^-1 * (num_speakers * smoothing_parameter^num_variables)^-2 ...
              * det(offender_covar)^-.5 * abs(det(offender_covar_inv + inv_kernel))^-.5 ...
              * dist_backspeakers_to_offender ...
              * det(suspect_covar)^-.5 * abs(det(suspect_covar_inv + inv_kernel))^-.5 ...
              * dist_backspeakers_to_suspect ;

likelihood_ratio = numerator / denominator;

% If the multiplictions in the numberator result in a number less than smallest floating point-number [2^(-1022) or about 2.2251e-308] 
% then the calculated likelihood ratio will be zero. To avoid this, set the output to the smallest floating-point number instead.
if likelihood_ratio == 0
    likelihood_ratio = realmin;
end

return

%% SUM OF SQUARED ERRORS FUNCTION

function y = sse(x)
% sum of squared errors calculation
[m] = size(x,1);
xc = bsxfun(@minus,x,sum(x,1)/m);  % Remove mean
y = xc' * xc;
return