% DATA PREPARATION

%load('C:\Users\Themis\Desktop\Thesis\DATASETS\BreaKHis\CPD\NONNEG_R40_BKH_40_033.mat','X','Y_binary','ID');
%load('C:\Users\Themis\Desktop\Thesis\DATASETS\BreaKHis\CPD\NONNEG_R40_BKH_100_033.mat','X','Y_binary','ID');
%load('C:\Users\Themis\Desktop\Thesis\DATASETS\BreaKHis\CPD\NONNEG_R40_BKH_200_033.mat','X','Y_binary','ID');
load('C:\Users\Themis\Desktop\Thesis\DATASETS\BreaKHis\CPD\NONNEG_R40_BKH_200_033.mat','X','Y_multiclass','ID');
%load('C:\Users\Themis\Desktop\Thesis\DATASETS\BreaKHis\CPD\NONNEG_R40_BKH_400_033.mat','X','Y_binary','ID');
%Y = Y_binary; clear Y_binary;
Y = Y_multiclass; clear Y_multiclass;
%load('C:\Users\Themis\Desktop\Thesis\DATASETS\UCSB-BCC\CPD\NONNEG_PROC_2_1_0_R40.mat','X','Y','ID');
%load('C:\Users\Themis\Desktop\Thesis\DATASETS\UCSB-BCC\CPD\NONNEG_PROC_5_1_0_R40.mat','X','Y','ID');
%load('C:\Users\Themis\Desktop\Thesis\DATASETS\UCSB-BCC\CPD\NONNEG_PROC_10_1_0_R40.mat','X','Y','ID');
%load('C:\Users\Themis\Desktop\Thesis\DATASETS\UCSB-BCC\CPD\NONNEG_PROC_5_1_0_R120.mat','X','Y','ID');

%X = double(X);
X = matrix_scale_columnwise(X,[0,1]);
%X = zscore(X);

% PARAMETERS

folds_k = 5;
folds_m = 5;
g_range = [0.1,100];
n_range = [0.001,0.6];
vr_range = [0.001,1];
bayes_reps = 300;
num_of_tries = 1;
param_sets = [13];

% TEST RUNS

TEST_RESULTS = cell(12,6);
TEST_RESULTS_ALL = cell(12*num_of_tries,6);
SCORES = zeros(12*num_of_tries,1);

for i = param_sets
    if i == 1
        num_hp = 3;
        cal_mode = 'sigmf';
        dec_mode = 'inf';
    elseif i == 2
        num_hp = 3;
        cal_mode = 'sigmf';
        dec_mode = 'exp';
    elseif i == 3
        num_hp = 3;
        cal_mode = 'sigmf';
        dec_mode = 'ent';
    elseif i == 4
        num_hp = 5;
        cal_mode = 'sigmf';
        dec_mode = 'inf';
    elseif i == 5
        num_hp = 5;
        cal_mode = 'sigmf';
        dec_mode = 'exp';
    elseif i == 6
        num_hp = 5;
        cal_mode = 'sigmf';
        dec_mode = 'ent';
    elseif i == 7
        num_hp = 3;
        cal_mode = 'evt';
        dec_mode = 'inf';
    elseif i == 8
        num_hp = 3;
        cal_mode = 'evt';
        dec_mode = 'exp';
    elseif i == 9
        num_hp = 3;
        cal_mode = 'evt';
        dec_mode = 'ent';
    elseif i == 10
        num_hp = 5;
        cal_mode = 'evt';
        dec_mode = 'inf';
    elseif i == 11
        num_hp = 5;
        cal_mode = 'evt';
        dec_mode = 'exp';
    elseif i == 12
        num_hp = 5;
        cal_mode = 'evt';
        dec_mode = 'ent';
    elseif i == 13
        num_hp = 17;
        cal_mode = 'evt';
        dec_mode = 'exp';
    end
    
    for j = 1:num_of_tries
        h_params = cv_optimise_h_params(X,Y,ID,num_hp,cal_mode,dec_mode,folds_k,folds_m,g_range,n_range,vr_range,bayes_reps);
        [CV_acc,CV_auc,v_acc,v_auc] = cv_validate_h_params(X,Y,ID,h_params,cal_mode,dec_mode,[],[]);
        close all
        
        TEST_RESULTS_ALL{(i-1)*num_of_tries+j,1} = CV_acc;
        TEST_RESULTS_ALL{(i-1)*num_of_tries+j,2} = std(v_acc);
        TEST_RESULTS_ALL{(i-1)*num_of_tries+j,3} = CV_auc;
        TEST_RESULTS_ALL{(i-1)*num_of_tries+j,4} = std(v_auc);
        TEST_RESULTS_ALL{(i-1)*num_of_tries+j,5} = h_params;
        TEST_RESULTS_ALL{(i-1)*num_of_tries+j,6} = [num2str(num_hp) ',' cal_mode ',' dec_mode ',result' num2str(j)];
        
        SCORES((i-1)*num_of_tries+j) = CV_acc - std(v_acc) + CV_auc - std(v_auc);
    end
    
    [~,max_ind] = max(SCORES(((i-1)*num_of_tries+1):(i*num_of_tries)));
    TEST_RESULTS(i,:) = TEST_RESULTS_ALL((i-1)*num_of_tries+max_ind,:);
    TEST_RESULTS{i,6} = extractBefore(TEST_RESULTS{i,6},",result");
end