% precision non-parametric
clear all
close all
addpath('.\m_files\', '.\m_files\cllr\');


% load data from each parrallel set of LRs
% prefix = {'GMM' 'b_GMM'};
prefix = {'MVKD2' 'b_MVKD2'};
for I_prefix = 1:2
    load(['.\results\', prefix{I_prefix}, '_fused.mat'], 'log_LR_fused', 'Indices_comparisons');
    
    % comparison-pair indices [assume they are idetical for each set
    II_ss = Indices_comparisons(:,1) == Indices_comparisons(:,2);
    II_ds = ~II_ss;

    log_LR_fused_ds(:, I_prefix) = log_LR_fused(II_ds);
    log_LR_fused_ss(:, I_prefix) = log_LR_fused(II_ss);
end

% convert to log10
log_LR_fused_ds = log_LR_fused_ds ./ log(10);


% get within-group mean and deviation from mean of each member of the group (there are 2 members in each group)
[mean_log_LR_fused_ds, sortII] = sort(mean(log_LR_fused_ds, 2));
log_LR_fused_ds = log_LR_fused_ds(sortII,:);
dev_from_mean_log_LR_fused_ds = log_LR_fused_ds - repmat(mean_log_LR_fused_ds, 1, 2);

% plot of mean logLR v deviance from mean
figure(1);
plot(mean_log_LR_fused_ds, dev_from_mean_log_LR_fused_ds(:,1), '.');
hold on
grid on
plot(mean_log_LR_fused_ds, dev_from_mean_log_LR_fused_ds(:,2), '.');
axis equal
xlabel('mean log10(LR)');
ylabel('deviation from mean');
hold off

% running nearest-neighbours 95% credible interval, successive linear regression to find 95% percentile
n = length(mean_log_LR_fused_ds);
num_nearest_neighbours = 80;
alpha = 0.05;
num_to_discard = num_nearest_neighbours * alpha;

hist_range = max(dev_from_mean_log_LR_fused_ds(:));
hist_edges = -hist_range : hist_range*2/50 : hist_range;
running_hist = NaN(51,n);

CI_half = NaN(n, 1);
for II = 1:n %n-400
    diff_from_mean_log_LR_fused_ds = mean_log_LR_fused_ds - mean_log_LR_fused_ds(II);
    [vals, II_diff] = sort(abs(diff_from_mean_log_LR_fused_ds));
    II_nearest = II_diff(1:num_nearest_neighbours);
    nearest_dev_from_mean_log_LR_fused_ds = dev_from_mean_log_LR_fused_ds(II_nearest,:);
    nearest_mean_log_LR_fused_ds = mean_log_LR_fused_ds(II_nearest,:);
    
    running_hist(:,II) = histc(nearest_dev_from_mean_log_LR_fused_ds(:), hist_edges);
    
    abs_half_nearest_dev_from_mean_log_LR_fused_ds = abs(nearest_dev_from_mean_log_LR_fused_ds(:,1));
    
    num_tokens_remaining = num_nearest_neighbours;
    y = abs_half_nearest_dev_from_mean_log_LR_fused_ds;
    x = nearest_mean_log_LR_fused_ds;
    
    while num_tokens_remaining > num_to_discard;
        % fit a linear regression
        intercept_col = ones(num_tokens_remaining, 1);
        [b, bint, residuals] = regress(y, [intercept_col, x]);
        [residuals_sorted, II_residuals_sorted] = sort(residuals);
        
        % throw away a portion of the data with the lowest signed residuals
        y = y(II_residuals_sorted);
        x = x(II_residuals_sorted);
        num_tokens_remaining = num_tokens_remaining - num_to_discard;
        
        y = y(num_to_discard + 1 : end);
        x = x(num_to_discard + 1 : end);

    end
    
    CI_half(II) = b(1) + b(2)*mean_log_LR_fused_ds(II);
    
end


% Waterfall plot of distribution
figure(3);
waterfall(hist_edges, mean_log_LR_fused_ds, running_hist');
axis tight
xlabel('deviation from mean log10(LR)');
ylabel('mean log10(LR)');
zlabel('count');
view([53 30]);
set(gca,'XDir','reverse');

% add CIs to scatterplot
figure(1);
hold on
plot(mean_log_LR_fused_ds, CI_half, '-r');
plot(mean_log_LR_fused_ds, -CI_half, '-r');
hold off

% Tippett plot with CI
mean_log_LR_fused_ss = mean(log_LR_fused_ss, 2);
plot_name = ['Fused mean LRs with 95% CI'];
plot_tippett([], [], 10.^(mean_log_LR_fused_ds - CI_half), [], [], true, '--', true, true);
plot_tippett([], [], 10.^(mean_log_LR_fused_ds + CI_half), [], [], false, '--', false, true);
plot_tippett(exp(mean_log_LR_fused_ss), [], 10.^mean_log_LR_fused_ds, [], plot_name, false, '-', false, false);

% Cllr of means
Cllr_means = cllr(mean_log_LR_fused_ss, mean_log_LR_fused_ds.*log(10));
Cllr_means

%clean up
rmpath('.\m_files\', '.\m_files\cllr\');
