function M_speaker = adaptUBM(UBM, x_tv, avoid_non_psd)
% Adapt UBM towards speaker model as per
% Reynold, Quatieri, Dunn (2000) Speaker verifiaction using adapted gaussian mixture models
%   Digital Signal Processing, 10, 19--41.

% I think this is equivalent to gmcluster line 241ff but using repmats, element by element multiplations, and sums, instead of loops and inner products
% post = Pr_ti
% S.PComponents = n_i
% S.mu = Ex_vi
% S.Sigma = Ex2_vi
% Xcentered.^2 = x2_tv
%   sort of: S.Sigma, Xcentered.^2 already include the homologue of eq 13; hence centred
%
% Should there be multiple iterations?
% Is strategy for avaoidance of non-psd valid?

w_i = UBM.PComponents;
mu_vi = UBM.mu';
mu2_vi = mu_vi.^2;
sigma2_vi = squeeze(UBM.Sigma); % diagonal covar

[T, V] = size(x_tv); %V is num vars

[idx, nlogl, Pr_ti] = cluster(UBM, x_tv);

n_i = sum(Pr_ti);

num_mixtures = length(n_i);

Pr_vit = permute(repmat(Pr_ti, [1 1 V]), [3 2 1]);
x_vit = permute(repmat(x_tv, [1 1 num_mixtures]), [2 3 1]);
n_vi = repmat(n_i, [V, 1]);

Ex_vi = sum(Pr_vit .* x_vit, 3) ./  n_vi; % S.mu(j,:) = post(:,j)' * X / S.PComponents(j);
Ex_vi(isnan(Ex_vi)) = 0;

x2_tv = x_tv.^2;%diag(x_tv * x_tv');
x2_vit = permute(repmat(x2_tv, [1 1 num_mixtures]), [2 3 1]);

Ex2_vi = sum(Pr_vit .* x2_vit, 3) ./  n_vi; % S.Sigma(:,:,j) = post(:,j)' * (Xcentered.^2) /S.PComponents(j) +regVal;
Ex2_vi(isnan(Ex2_vi)) = 0;


r = 16;
alpha_i = n_i ./ (n_i + r);
alpha_vi = repmat(alpha_i, V, 1);

w_hat_i_temp = alpha_i .* n_i / T + (1 - alpha_i) .* w_i;
w_hat_i = w_hat_i_temp / sum(w_hat_i_temp);

mu_hat_vi = alpha_vi .* Ex_vi + (1 - alpha_vi) .* mu_vi;

mu2_hat_vi = mu_hat_vi.^2;
sigma2_hat_vi = alpha_vi .* Ex2_vi + (1 - alpha_vi) .* (sigma2_vi + mu2_vi) - mu2_hat_vi;
% avoid non pos-semi-def
sigma2_hat_vi(sigma2_hat_vi(:) < avoid_non_psd) = avoid_non_psd;
for j = 1:num_mixtures
    min_allowed = max(sigma2_hat_vi(:,j)) * eps;
    sigma2_hat_vi(sigma2_hat_vi(:,j) < min_allowed, j) = min_allowed;
end

mu_hat_iv = mu_hat_vi';
sigma2_hat_1vi = permute(sigma2_hat_vi, [3 1 2]);
M_speaker = gmdistribution(mu_hat_iv, sigma2_hat_1vi, w_hat_i);

end

