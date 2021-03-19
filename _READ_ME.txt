© 2010 Geoffrey Stewart Morrison
http://geoff-morrison.net
2010-02-19

This folder contains data, results, and Matlab scripts and fuctions for analyses reported in:
	Morrison, G. S. (2010) A comparison of procedures for the calculation of forensic likelihood ratios from acoustic-phonetic data: Multvariate kernel density (MVKD) versus Gaussian mixture model – universal background model (GMM-UBM)


The software has been tested using Matlab R2009b running under Windows XP x64.

Cllr and logistic-regression fusion functions (and associated functions) from Niko Brümmer's FoCal Tookit <http://sites.google.com/site/nikobrummer/focal> must be in the following folders (or make approprate changes if they are elsewhere):
	.\m_files\cllr\
	.\m_files\fusion\

Morrison's robust version of the fusion training function is included as:
	.\m_files\train_llr_fusion_robust.m 


To generate the A and B sets, use:
	run_MVKD2_likelihood_ratio_A.m
	run_MVKD2_likelihood_ratio_B.m
	run_GMM_likelihood_ratio_A.m
	run_GMM_likelihood_ratio_B_limited.m

To generate the pooled GMM-UBM (A-set) results, use:
	run_GMM_likelihood_ratio_A_pooled.m

To find best the combination of num GMMs and num iterations in GMM-UBM (A-set) results (including plotting Cllr against num GMMs and num itereations), use:
	find_best.m

To fuse the results from different phonemes, use:
	run_fuse_MVKD_A.m
	run_fuse_MVKD_B.m
	run_fuse_best_GMM_A.m
	run_fuse_best_GMM_B_limited.m

To estimate credible interval for LR output, use:
	precision_non_parametric.m
