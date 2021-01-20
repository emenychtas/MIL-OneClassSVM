function [cv_error] = cv_bayesopt_fun(X,Y,ID,cv_folds,h_params,cal_mode,dec_mode)
    % This is an objective function used by the bayesian optimiser in order
    % to tweak the hyper-parameters for the mil_one_class_svm function. What
    % it does is execute mil_one_class_svm for many folds given a single set
    % of hyper-parameters and return a cv_error value according to a rule.
    %
    % Usage:
    %
    % [cv_error] = cv_bayesopt_fun(X,Y,ID,cv_folds,dec_mode,h_params);
    %
    % OR
    %
    % a = optimizableVariable('a',aRange);
    % b = optimizableVariable('b',bRange);
    % fun = @(x)cv_bayesopt_fun(X,Y,ID,cv_folds,dec_mode,[x.a x.b]);
    % results = bayesopt(fun,[a,b],'UseParallel',true,'MaxObjectiveEvaluations',30);
    
    for i = 1:length(cv_folds)
        try
            [v_acc(i),v_auc(i)] = mil_one_class_svm(X,Y,ID,cv_folds{i},h_params,cal_mode,dec_mode,[],[]);
        catch ME
            v_acc(i) = 0;
            v_auc(i) = 0;
        end
    end
    
    cv_error = 1 - mean(v_acc);
    %cv_error = 1 - mean(v_auc);
    %cv_error = - mean(v_acc) + std(v_acc) - mean(v_auc) + std(v_auc);
end