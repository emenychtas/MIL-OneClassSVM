function [acc,auc] = mil_one_class_svm(X,Y,ID,data_sets,h_params,cal_mode,dec_mode,T,bag_props)
    % This function trains a one-class SVM per class based on the matrices X,Y,ID,
    % and h_params. These matrices are broken down to parts based on the data_sets struct
    % which contains the bag IDs of a training set and a validation set. After that
    % each SVM makes a prediction for every instance and a complex algorithm makes a
    % decision for the label of each bag. There are 3 decision modes:
    %
    % 0 -> mean + informative windows
    % 1 -> mean only
    % 2 -> entropy only
    %
    % In case informative windows are used, this function can optionally display them
    % using tensor T and struct bag_props from the input arguments. These arguments
    % should be given as [] if no display is desired.
    %
    % Usage:
    %
    % [acc,auc] = ...
    %    mil_one_class_svm(X,Y,ID,data_sets,h_params,dec_mode,T,bag_props);
    
    %% DATA PREPARATION
    
    nr_of_classes = length(unique(Y));
    class_prior_probs = zeros(nr_of_classes,1);
    
    % Check if EVT tail size is given along with PCA variance retained.
    if mod(length(h_params),2) == 1
        X = matrix_pca_variance_retained(X,h_params(end));
        tail_size = -1;
    else
        X = matrix_pca_variance_retained(X,h_params(end-1));
        tail_size = h_params(end);
    end
    
    %X = matrix_scale_columnwise(X,[0,1]);
    %X = zscore(X);
    
    % Check if single set of SVM hyper-parameters is given.
    if length(h_params)<5
        temp = h_params(1:2);
        h_params = [];
        for i = 1:nr_of_classes
            h_params = horzcat(h_params,temp);
        end
    end
    
    % Create separate SVM training data and calculate prior class probabilities.
    X_train = cell(nr_of_classes,1);
    for i = 1:length(data_sets.tr)
        temp_index = unique(Y(ID==data_sets.tr(i))) + 1;
        X_train{temp_index} = vertcat(X_train{temp_index},X(ID==data_sets.tr(i),:,:));
        class_prior_probs(temp_index) = class_prior_probs(temp_index) + 1/length(data_sets.tr);
    end
    
    % Create validation data.
    X_val = [];
    Y_val = [];
    ID_val = [];
    for i = 1:length(data_sets.val)
        X_val = vertcat(X_val,X(ID==data_sets.val(i),:,:));
        Y_val = vertcat(Y_val,Y(ID==data_sets.val(i)));
        ID_val = vertcat(ID_val,ID(ID==data_sets.val(i)));
    end
    
    ID_val_unique = unique(ID_val);
    nr_of_val_subjects = length(ID_val_unique);
    
    % Logistic Regression
    %lr_X = vertcat(X_train{1},X_train{2});
    %lr_Y = vertcat(ones(size(X_train{1},1),1),2*ones(size(X_train{2},1),1));
    
    %% SVM TRAINING
    
    svm_models = cell(nr_of_classes,1);
    train_accuracy = cell(nr_of_classes,1);
    train_dec_values = cell(nr_of_classes,1);
    train_pred_labels = cell(nr_of_classes,1);
    
    % Train one-class SVMs only with their own instances from the training set.
    parfor i = 1:nr_of_classes
        % Labels for class i. All positive.
        Y_train_i = ones(size(X_train{i},1),1);
        svm_options{i} = ['-s 2 -t 2 -h 0 -m 2000 -g ' num2str(h_params((i-1)*2+1)) ' -n ' num2str(h_params((i-1)*2+2))];
        
        [svm_models{i}] = suppressed_svmtrain(Y_train_i,X_train{i},svm_options{i});
        [train_pred_labels{i},train_accuracy{i},train_dec_values{i}] = suppressed_svmpredict(Y_train_i,X_train{i},svm_models{i});
        
        % Logistic Regression
        %[train_pred_labels{i},train_accuracy{i},train_dec_values{i}] = suppressed_svmpredict(lr_Y,lr_X,svm_models{i});
    end
    
    % Fit a Weibull PDF to a subset of each SVM's training decision values.
    W = svm_weibull_fit(train_dec_values,svm_models,tail_size);
    
    % Logistic Regression
    %B{1} = mnrfit(train_dec_values{1},lr_Y);
    %B{2} = mnrfit(train_dec_values{2},lr_Y);
    
    %% SVM PREDICTION
    
    val_accuracy = cell(nr_of_classes,1);
    val_dec_values = zeros(size(X_val,1),nr_of_classes);
    val_pred_labels = zeros(size(X_val,1),nr_of_classes);
    
    % Validate one-class SVMs using all instances from the validation set.
    parfor i = 1:nr_of_classes
        % Labels for class i. Own 1, others -1.
        Y_val_i = (Y_val==(i-1)) - (Y_val~=(i-1));
        
        [val_pred_labels(:,i),val_accuracy{i},val_dec_values(:,i)] = suppressed_svmpredict(Y_val_i,X_val,svm_models{i});
    end
    
    % Calibrate SVM decision values.
    if isequal(cal_mode,'sigmf')
        % Use a simple logistic function.
        val_dec_values = sigmf(val_dec_values,[1 0]);
    elseif isequal(cal_mode,'evt')
        % Project each SVM's validation decision values onto the corresponding
        % Weibull CDF to get unnormalized posterior probabilities which are not
        % true probabilites but are comparable to each other.
        val_dec_values = svm_weibull_trans(val_dec_values,W,class_prior_probs,'open');
    elseif isequal(cal_mode,'lr')
        % Logistic Regression
        %lr_temp = mnrval(B{1},val_dec_values(:,1));
        %val_dec_values(:,1) = lr_temp(:,1);
        %lr_temp = mnrval(B{2},val_dec_values(:,2));
        %val_dec_values(:,2) = lr_temp(:,2);
    end
    
    % Plot the Weibull CDFs used for calibration.
    %for i = 1:nr_of_classes
        %figure; fplot(@(x) wblcdf(x,W(i,1),W(i,2)),[-5,20]);
    %end
    
    %% SUBJECT CLASSIFICATION
    
    subj_pred_labels = zeros(nr_of_val_subjects,1);
    
    % Decision values needed for AUC (binary only).
    if nr_of_classes == 2
        subj_dec_values = zeros(nr_of_val_subjects,1);
    end
    
    for i = 1:nr_of_val_subjects
        %% PHASE ONE - BASIC EVALUATION
        
        % Get the decision values of all one-class SVMs for subject i's instances.
        subj_i_instance_dec_values = val_dec_values(ID_val==ID_val_unique(i),:);
        
        % Predict subject i's label based on the max score.
        if isequal(dec_mode,'inf') || isequal(dec_mode,'exp')
            % Mean per SVM.
            subj_i_scores = mean(subj_i_instance_dec_values,1);
        elseif isequal(dec_mode,'ent')
            % Entropy per SVM.
            subj_i_scores = subj_i_instance_dec_values.*log(subj_i_instance_dec_values);
            subj_i_scores(isnan(subj_i_scores)) = 0;
            subj_i_scores = - sum(subj_i_scores,1);
        end
        [~,temp_index] = max(subj_i_scores);
        subj_pred_labels(i) = temp_index - 1;
        
        % Decision values needed for AUC (binary only).
        if nr_of_classes == 2
            subj_dec_values(i) = subj_i_scores(2) - subj_i_scores(1);
        end
        
        % Plot subject i's decision values per SVM.
        %figure('Name',['Subject #' num2str(ID_val_unique(i)) ' - Malignant? ' num2str(unique(Y_val(ID_val==ID_val_unique(i))))]);
        %plot(subj_i_instance_dec_values);
        
        %% PHASE TWO - RE-EVALUATION (INFORMATIVE WINDOWS)
        
        if isequal(dec_mode,'inf')
            % Get the predicted labels of all one-class SVMs for subject i's instances.
            subj_i_instance_pred_labels = val_pred_labels(ID_val==ID_val_unique(i),:);
            
            % Sum predicted labels by rows and compare with the informative sum (only one 1).
            informative_windows = (sum(subj_i_instance_pred_labels,2) == (nr_of_classes-1)*(-1)+1);
            
            if nnz(informative_windows) == 0
                disp([' Subject #' num2str(ID_val_unique(i)) ': No informative windows found.']);
            else
                % Predict subject i's label once more based on the max score. Mean per SVM.
                subj_i_scores = mean(subj_i_instance_dec_values(informative_windows,:),1);
                [max_score,temp_index] = max(subj_i_scores);
                subj_pred_labels(i) = temp_index - 1;
                
                % Decision values needed for AUC (binary only).
                if nr_of_classes == 2
                    subj_dec_values(i) = subj_i_scores(2) - subj_i_scores(1);
                end
                
                % Check for ambiguous classifiaction.
                if sum(subj_i_scores==max_score) > 1
                    disp([' Subject #' num2str(ID_val_unique(i)) ': ' num2str(nnz(informative_windows)) ' informative windows found, but ambiguous.']);
                else
                    disp([' Subject #' num2str(ID_val_unique(i)) ': ' num2str(nnz(informative_windows)) ' informative windows found.']);
                end
                
                % Display informative windows.
                if ~isempty(T)
                    indices = find(ID==ID_val_unique(i));
                    for j = 1:length(indices)
                        if informative_windows(j) == 0
                            indices(j) = nan;
                        end
                    end
                    patches_to_bag(T(indices(~isnan(indices)),:,:),ID(indices(~isnan(indices))),bag_props,ID_val_unique(i));
                end
            end
        end
    end
    
    %% ACCURACY AND AUC CALCULATION
    
    % Calculate the correct subject labels.
    correct_labels = zeros(nr_of_val_subjects,1);
    for i = 1:nr_of_val_subjects
        correct_labels(i) = unique(Y_val(ID_val==ID_val_unique(i)));
    end
    
    % Calculate subject classification test accuracy.
    acc = 0;
    for i = 1:nr_of_val_subjects
        if subj_pred_labels(i) == correct_labels(i)
            acc = acc + 1;
        else
            disp([' Subject #' num2str(ID_val_unique(i)) ': Wrong label.']);
        end
    end
    acc = acc / nr_of_val_subjects;
    
    % Calculate AUC if classification problem is binary.
    if nr_of_classes == 2
        [~,~,~,auc] = perfcurve(correct_labels,subj_dec_values,1);
    else
        auc = nan;
    end
end