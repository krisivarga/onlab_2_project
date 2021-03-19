function w = train_llr_fusion_robust(targets, non_targets, prior, robust_weight)
%
% Revised version by Geoffrey Stewart Morrison 2009-07-02 which handles cases of complete separation in the data.
% Mislabels one copy of 'targets' as non-targets, and gives it a weight of 'robust_weight' times the correctly labelled 'non_targets'.
% Mutatis mutandis for one copy of 'non-targets'.
% A suitable value for 'robust_weight' might be 0.001
% Default value for 'robust_weight' is 0, i.e., same as original 'train_llr_fusion' function
% 
%
%  Train Linear fusion with prior-weighted Logistic Regression objective.
%  The fusion output is encouraged by this objective to be a well-calibrated log-likelihood-ratio.
%  I.E., this is simultaneous fusion and calibration.
%
%  Usage:
%    w = TRAIN_LLR_FUSION(targets,non_targets,prior)
%    w = TRAIN_LLR_FUSION(targets,non_targets)
%   
%  Input parameters:
%    targets:     a [d,nt] matrix of nt target scores for each of d systems to be fused. 
%    non_targets: a [d,nn] matrix of nn non-target scores for each of the d systems.
%    prior:       (optional, default = 0.5), a scalar parameter between 0 and 1. 
%                 This weights the objective function, by replacing the effect 
%                 that the proportion nt/(nn+nt) has on the objective.
%                 For general use, omit this parameter (i.e. prior = 0.5).
%                 For NIST SRE, use: prior = effective_prior(0.01,10,1);
%
%  Output parameters:
%    w: a vector of d+1 fusion coefficients. 
%         The first d coefficients are the weights for the d input systems.
%         The last coefficient is an offset (see below).
%
%  Fusion of new scores:
%    see: LIN_FUSION.m
%                                 
%

%  This code is an adapted version of the m-file 'train_cg.m' as made available by Tom Minka
%  at http://www.stat.cmu.edu/~minka/papers/logreg/.

max_iter = 5000; %originally hard coded as 1000

if nargin < 3 || isempty(prior)
   prior = 0.5;
end


nt = size(targets,2);
nn = size(non_targets,2);

% original:
% prop = nt/(nn+nt);
% weights = [(prior/prop)*ones(1,nt),((1-prior)/(1-prop))*ones(1,nn)];
% x = [[targets;ones(1,nt)],-[non_targets;ones(1,nn)]];
% offset = logit(prior)*[ones(1,nt),-ones(1,nn)];

% make robust to complete separation
if nargin < 4 || isempty(robust_weight)
    robust_weight = 0;
end

ntnn = nt+nn;

x = [[targets, non_targets; ones(1,ntnn)],-[non_targets, targets; ones(1,ntnn)]];

robust_weight_targets = nt/nn * robust_weight;
robust_weight_non_targets = nn/nt * robust_weight;

prop_targets = nt / ntnn;
prop_non_targets = 1 - prop_targets;

prior_adjusted_targets = prior/prop_targets;
prior_adjusted_non_targets = (1-prior)/prop_non_targets;

weights = [prior_adjusted_targets*ones(1,nt), prior_adjusted_targets*robust_weight_targets*ones(1,nn), prior_adjusted_non_targets*ones(1,nn), prior_adjusted_non_targets*robust_weight_non_targets*ones(1,nt)];

offset = log(prior/(1-prior))*[ones(1,ntnn),-ones(1,ntnn)];





w = zeros(size(x,1),1);

[d,n] = size(x);
old_g = zeros(size(w));
for iter = 1:max_iter
  old_w = w;
  % s1 = 1-sigma
  s1 = 1./(1+exp(w'*x+offset));
  g = x*(s1.*weights)';
  if iter == 1
    u = g;
  else
    u = cg_dir(u, g, old_g);
  end
  
  % line search along u
  ug = u'*g;
  ux = u'*x;
  a = weights.*s1.*(1-s1);
  uhu = (ux.^2)*a';
  w = w + (ug/uhu)*u;
  old_g = g;
  if max(abs(w - old_w)) < 1e-5 % originally coded as 1e-5
     break
  end
end
if iter == max_iter
  warning('not enough iters')
end
