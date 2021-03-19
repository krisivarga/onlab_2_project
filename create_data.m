% feats_norm = feat2;
% for i=1:size(feats_norm,2)
%     feats_norm(:,i) = feats_norm(:,i)/max(feats_norm(:,i));
% end
% numdim = 1;
% background_data = feats_norm(1:100,:);
% background_index = spk_ids(1:100);
% suspect_data = feats_norm(101:2:end,:);
% offender_data = feats_norm(102:2:end,:);
% offender_index = spk_ids(102:2:end);
% suspect_index = spk_ids(101:2:end);
% likelihood_ratio = multivar_kernel_LR(suspect_data(5,:), offender_data(5,:), background_data, background_index)
% 
%------------------------------------------------------------------------------------------------------------------------


%-------------------------------önálló "munka"------------------------
%seassion 1 és 2 szétválogatás

load('.\data\forvoice_data2.mat');
session_1_spk_ids = spk_ids(session_ids(:,1) == 1 & task_ids(:,1) ~= 4, :);
session_1_feat2 = feat2(session_ids(:,1) == 1 & task_ids(:,1) ~= 4, :);
session_2_spk_ids = spk_ids(session_ids(:,1) == 2 & task_ids(:,1) ~= 4, :);
session_2_feat2 = feat2(session_ids(:,1) == 2 & task_ids(:,1) ~= 4, :);
results=[];


for i=1:10
%random
k1 = randperm(size(session_1_spk_ids,1));
k2 = randperm(size(session_2_spk_ids,1));

%első 60 sesion1-ből, maradék 20-ből az első 10
%session2-ből az első 10
% train_data = data(k1(1:180), :);
% enroll_data = data(k1(181:190), :);
% test_data = data(k2(1:10), :);

feats_norm = feat2;
% for i=1:size(feats_norm,2)
%     feats_norm(:,i) = feats_norm(:,i)/max(feats_norm(:,i));
% end
% numdim = 1;
background_data = feats_norm(k1(1:180), :);
background_index = spk_ids(k1(1:180));

suspect_data = feats_norm(k1(181:190), :);
suspect_index = spk_ids(k1(181:190));

offender_data = feats_norm(k2(1:10), :);
offender_index = spk_ids(k2(1:10));



    likelihood_ratio = multivar_kernel_LR(suspect_data(5,:), offender_data(5,:), background_data, background_index)
    results(end+1) = likelihood_ratio;
end

mean(results,2)
