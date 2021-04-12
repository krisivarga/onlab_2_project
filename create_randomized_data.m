% %ami kell  // 60-20-ra osztva
% %1) beszél? azonosítók 
% %2) session azonositók
% %3) a hozzá tartozó adat
% 
% %4) kell egy olyan .mat fájl, amiben van egy 60 beszél?t tartalmazó mind 3
% %fejladatra összemosott (e-h) 1-3 session 1-2
% % 20 beszél?re 1-3 session 1re!!
% % 20 beszél?re 1-3 session 2re!!
% %5) kell 3 .mat fájl mint 4, de mind a 3 formantra
% 
% %pl:  (73) (1)  (176.7433668	1.047511711	-0.458693506	0.536145095)
% 
% 

%adatok beolv
T = readtable('features.csv');
%egyedi speakerIDs
 speakerIDs = unique(T(:,[2]));
% random szétválogatás 60-20ra
 Indices=randperm(80);
 speakerIDsTrainingset=table2array(speakerIDs(Indices(1:60),:));
 speakerIDsTestingset=table2array(speakerIDs(Indices(61:end),:));

% create data tables for selection 
data_all = table2array(T(:,[2:3 5:8]));
data_formant1 = table2array(T(:,[2:3 9:12]));
data_formant2 = table2array(T(:,[2:3 17:20]));
data_formant3 = table2array(T(:,[2:3 25:28]));


%getVarName(data_all)

%all_test
create_data_from(data_all,speakerIDsTestingset,3,getVarName(data_all),'test')
create_data_from(data_all,speakerIDsTestingset,1,getVarName(data_all),'test')
create_data_from(data_all,speakerIDsTestingset,2,getVarName(data_all),'test')

%all_train
create_data_from(data_all,speakerIDsTrainingset,3,getVarName(data_all),'train')
create_data_from(data_all,speakerIDsTrainingset,1,getVarName(data_all),'train')
create_data_from(data_all,speakerIDsTrainingset,2,getVarName(data_all),'train')



%formant1_test
create_data_from(data_formant1,speakerIDsTestingset,3,getVarName(data_formant1),'test')
create_data_from(data_formant1,speakerIDsTestingset,1,getVarName(data_formant1),'test')
create_data_from(data_formant1,speakerIDsTestingset,2,getVarName(data_formant1),'test')

%formant1_train
create_data_from(data_formant1,speakerIDsTrainingset,3,getVarName(data_formant1),'train')
create_data_from(data_formant1,speakerIDsTrainingset,1,getVarName(data_formant1),'train')
create_data_from(data_formant1,speakerIDsTrainingset,2,getVarName(data_formant1),'train')


%formant2_test
create_data_from(data_formant2,speakerIDsTestingset,3,getVarName(data_formant2),'test')
create_data_from(data_formant2,speakerIDsTestingset,1,getVarName(data_formant2),'test')
create_data_from(data_formant2,speakerIDsTestingset,2,getVarName(data_formant2),'test')

%formant2_train
create_data_from(data_formant2,speakerIDsTrainingset,3,getVarName(data_formant2),'train')
create_data_from(data_formant2,speakerIDsTrainingset,1,getVarName(data_formant2),'train')
create_data_from(data_formant2,speakerIDsTrainingset,2,getVarName(data_formant2),'train')

%formant3_test
create_data_from(data_formant3,speakerIDsTestingset,3,getVarName(data_formant3),'test')
create_data_from(data_formant3,speakerIDsTestingset,1,getVarName(data_formant3),'test')
create_data_from(data_formant3,speakerIDsTestingset,2,getVarName(data_formant3),'test')

%formant3_train
create_data_from(data_formant3,speakerIDsTrainingset,3,getVarName(data_formant3),'train')
create_data_from(data_formant3,speakerIDsTrainingset,1,getVarName(data_formant3),'train')
create_data_from(data_formant3,speakerIDsTrainingset,2,getVarName(data_formant3),'train')



function f = create_data_from(select_data_from,select_data_by,session,name,type)
    selected_all_cols = [];
    if session == 1 || session == 2
        for i=1:length(select_data_by)
            selected_rows = select_data_from( select_data_from(:,1)==select_data_by(i) & select_data_from(:,2) == session , :);
            selected_all_cols = [selected_all_cols;selected_rows];
        end
    else
        for i=1:length(select_data_by)
            selected_rows = select_data_from( select_data_from(:,1)==select_data_by(i) & (select_data_from(:,2) == 1 | select_data_from(:,2) == 2 ) , :);
            selected_all_cols = [selected_all_cols;selected_rows];
        end
    end
    
        Indices_Speakers = selected_all_cols(:,[1]);
        Indices_Sessions = selected_all_cols(:,[2]);
        data = selected_all_cols(:,[3:6]);
        
        filename=sprintf('%s.mat',name);
        %vars = {'Indices_Speakers' 'Indices_Sessions' 'data'};
        save(['data_rand/session_',num2str(session),'_',type,'_',filename],'Indices_Speakers','Indices_Sessions','data')
end

function out = getVarName(var)
    out = inputname(1);
end






