function [cv_folds]  = cv_kfold_stratified_sets(Y,ID,k)
    % This function takes matrices Y and ID containing the label and bag ID
    % of every instance and the integer k. The output is a cell of k structs
    % that contain a training set and a validation set. These sets are just
    % vectors of bag ID's, they don't contain the instances themselves.
    %
    % Usage:
    %
    % [cv_folds]  = cv_kfold_stratified_sets(Y,ID,k);
    
    % Set the seed for reproducibility.
    rng(0,'twister');
    
    % Find unique subjects.
    ID_unique = unique(ID);
    nrOfSubjects = length(ID_unique);
    
    % Find the label of each subject.
    subj_labels = zeros(nrOfSubjects,1);
    for i = 1:nrOfSubjects
        subj_labels(i) = Y(find(ID==ID_unique(i),1));
    end
    
    % Create cross-validation folds.
    cv_folds = cell(k,1);
    c = cvpartition(subj_labels,'KFold',k);
    for i = 1:k
        cv_folds{i}.tr = ID_unique(c.training(i));
        cv_folds{i}.val = ID_unique(c.test(i));
    end
end