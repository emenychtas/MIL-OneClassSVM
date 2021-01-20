function [varargout] = cv_validate_h_params(X,Y,ID,h_params,cal_mode,dec_mode,T,bag_props,varargin)
    % This function is used to test the hyper-parameters already obtained by
    % the cv_optimise_h_params function. It either tests many folds or a single
    % fold, depending on the syntax.
    %
    % Usage:
    %
    % [acc,auc] = ...
    %     cv_validate_h_params(X,Y,ID,h_params,dec_mode,T,bag_props,i);
    % [CV_acc,CV_auc,v_acc,v_auc] = ...
    %     cv_validate_h_params(X,Y,ID,h_params,dec_mode,T,bag_props);
    %
    % Note:
    %
    % In the first syntax, i is the single fold to test.
    
    % Check which version of the function to use.
    if (nargout==2 && nargin==9)
        % Reproduce dataset folds.
        [cv_folds] = cv_kfold_stratified_sets(Y,ID,size(h_params,1));
        
        % Obtain the accuracy and area-under-curve metrics.
        [varargout{1},varargout{2}] = ...
            mil_one_class_svm(X,Y,ID,cv_folds{varargin{1}},h_params(varargin{1},:),cal_mode,dec_mode,T,bag_props);
    elseif (nargout==4 && nargin==8)
        % Reproduce dataset folds.
        [cv_folds] = cv_kfold_stratified_sets(Y,ID,size(h_params,1));
        
        % Obtain the accuracy and area-under-curve metrics for every fold.
        for i = 1:size(h_params,1)
            [varargout{3}(i),varargout{4}(i)] = ...
                mil_one_class_svm(X,Y,ID,cv_folds{i},h_params(i,:),cal_mode,dec_mode,T,bag_props);
        end
        
        % Calculate the mean of each metric.
        varargout{1} = mean(varargout{3});
        varargout{2} = mean(varargout{4});
    else
        disp('Invalid input/output arguments.');
    end
end