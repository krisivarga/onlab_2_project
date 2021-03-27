data_all_raw = readtable('features_torolt.csv');

for c= 1:4
    Indices_Session = [data_all_raw(:,3)];
    indices_Speakers = [data_all_raw(:,2)];
    data = [data_all_raw(:,4+(c-1)*4:4+3+(c-1)*4)];
    
    save(['data_' int2str(c) '.mat'],'Indices_Session','indices_Speakers','data')
end