function [h_params] = cv_optimise_h_params(X,Y,ID,num_hp,cal_mode,dec_mode,k,m,g_range,n_range,vr_range,bayes_reps,varargin)
    % This function makes k "outer" folds from the dataset, and then another
    % m "inner" folds from the training set of each "outer" fold. After doing
    % so, it runs a bayesian optimiser with the given parameters on the m "inner"
    % folds and outputs the hyper-parameters that minimise the objective function
    % for each of the k "outer" folds. Those can be then used with cv_validate_h_params
    % to obtain some cross-validation metrics.
    %
    % Usage:
    %
    % [h_params] = cv_optimise_h_params(X,Y,ID,dec_mode,k,m,g_range,n_range,bayes_reps,varargin);
    %
    % Note:
    %
    % If varargin is set to an index, this function optimises the h_params only for the given fold.
    
    % Produce k "outer" folds.
    [k_cv_folds] = cv_kfold_stratified_sets(Y,ID,k);
    
    % Check whether to optimise one or all folds.
    if isempty(varargin)
        a = 1;
        b = k;
    else
        a = varargin{1};
        b = a;
    end
    
    for i = a:b
        % For each "outer" fold reconstruct matrices Y and ID of the
        % training set because they're needed for cv_kfold_stratified_sets.
        Y_i = [];
        ID_i = [];
        for j = 1:length(k_cv_folds{i}.tr)
            Y_i = vertcat(Y_i,Y(ID==k_cv_folds{i}.tr(j)));
            ID_i = vertcat(ID_i,ID(ID==k_cv_folds{i}.tr(j)));
        end
        
        % Produce m "inner" folds from the training set of each "outer" fold.
        [m_cv_folds] = cv_kfold_stratified_sets(Y_i,ID_i,m);
        
        % Prepare and run the bayesian optimiser.
        if num_hp == 3
            if size(g_range,1) == 1
                g1 = optimizableVariable('g1',g_range);
            else
                g1 = optimizableVariable('g1',g_range(1,:));
            end
            n1 = optimizableVariable('n1',n_range);
            vr = optimizableVariable('vr',vr_range);
            
            fun = @(x)cv_bayesopt_fun(X,Y,ID,m_cv_folds,[x.g1 x.n1 x.vr],cal_mode,dec_mode);
            results = bayesopt(fun,[g1,n1,vr],'UseParallel',true,'MaxObjectiveEvaluations',bayes_reps);
            
            % Return best (estimated) hyper-parameters along with their measured error.
            h_params(i,1) = results.XAtMinEstimatedObjective.g1;
            h_params(i,2) = results.XAtMinEstimatedObjective.n1;
            h_params(i,3) = results.XAtMinEstimatedObjective.vr;
        elseif num_hp == 4
            if size(g_range,1) == 1
                g1 = optimizableVariable('g1',g_range);
            else
                g1 = optimizableVariable('g1',g_range(1,:));
            end
            n1 = optimizableVariable('n1',n_range);
            vr = optimizableVariable('vr',vr_range);
            ts = optimizableVariable('ts',[0,1]);
            
            fun = @(x)cv_bayesopt_fun(X,Y,ID,m_cv_folds,[x.g1 x.n1 x.vr x.ts],cal_mode,dec_mode);
            results = bayesopt(fun,[g1,n1,vr,ts],'UseParallel',true,'MaxObjectiveEvaluations',bayes_reps);
            
            % Return best (estimated) hyper-parameters along with their measured error.
            h_params(i,1) = results.XAtMinEstimatedObjective.g1;
            h_params(i,2) = results.XAtMinEstimatedObjective.n1;
            h_params(i,3) = results.XAtMinEstimatedObjective.vr;
            h_params(i,4) = results.XAtMinEstimatedObjective.ts;
        elseif num_hp == 5
            if size(g_range,1) == 1
                g1 = optimizableVariable('g1',g_range);
                g2 = optimizableVariable('g2',g_range);
            else
                g1 = optimizableVariable('g1',g_range(1,:));
                g2 = optimizableVariable('g2',g_range(2,:));
            end
            n1 = optimizableVariable('n1',n_range);
            n2 = optimizableVariable('n2',n_range);
            vr = optimizableVariable('vr',vr_range);
            
            fun = @(x)cv_bayesopt_fun(X,Y,ID,m_cv_folds,[x.g1 x.n1 x.g2 x.n2 x.vr],cal_mode,dec_mode);
            results = bayesopt(fun,[g1,n1,g2,n2,vr],'UseParallel',true,'MaxObjectiveEvaluations',bayes_reps);
            
            % Return best (estimated) hyper-parameters along with their measured error.
            h_params(i,1) = results.XAtMinEstimatedObjective.g1;
            h_params(i,2) = results.XAtMinEstimatedObjective.n1;
            h_params(i,3) = results.XAtMinEstimatedObjective.g2;
            h_params(i,4) = results.XAtMinEstimatedObjective.n2;
            h_params(i,5) = results.XAtMinEstimatedObjective.vr;
        elseif num_hp == 6
            if size(g_range,1) == 1
                g1 = optimizableVariable('g1',g_range);
                g2 = optimizableVariable('g2',g_range);
            else
                g1 = optimizableVariable('g1',g_range(1,:));
                g2 = optimizableVariable('g2',g_range(2,:));
            end
            n1 = optimizableVariable('n1',n_range);
            n2 = optimizableVariable('n2',n_range);
            vr = optimizableVariable('vr',vr_range);
            ts = optimizableVariable('ts',[0,1]);
            
            fun = @(x)cv_bayesopt_fun(X,Y,ID,m_cv_folds,[x.g1 x.n1 x.g2 x.n2 x.vr x.ts],cal_mode,dec_mode);
            results = bayesopt(fun,[g1,n1,g2,n2,vr,ts],'UseParallel',true,'MaxObjectiveEvaluations',bayes_reps);
            
            % Return best (estimated) hyper-parameters along with their measured error.
            h_params(i,1) = results.XAtMinEstimatedObjective.g1;
            h_params(i,2) = results.XAtMinEstimatedObjective.n1;
            h_params(i,3) = results.XAtMinEstimatedObjective.g2;
            h_params(i,4) = results.XAtMinEstimatedObjective.n2;
            h_params(i,5) = results.XAtMinEstimatedObjective.vr;
            h_params(i,6) = results.XAtMinEstimatedObjective.ts;
        elseif num_hp == 17
            
            
            
            %%%% EXPERIMENTAL %%%%
            
            
            
            if size(g_range,1) == 1
                g1 = optimizableVariable('g1',g_range);
                g2 = optimizableVariable('g2',g_range);
                g3 = optimizableVariable('g3',g_range);
                g4 = optimizableVariable('g4',g_range);
                g5 = optimizableVariable('g5',g_range);
                g6 = optimizableVariable('g6',g_range);
                g7 = optimizableVariable('g7',g_range);
                g8 = optimizableVariable('g8',g_range);
            else
                g1 = optimizableVariable('g1',g_range(1,:));
                g2 = optimizableVariable('g2',g_range(2,:));
                g3 = optimizableVariable('g3',g_range(3,:));
                g4 = optimizableVariable('g4',g_range(4,:));
                g5 = optimizableVariable('g5',g_range(5,:));
                g6 = optimizableVariable('g6',g_range(6,:));
                g7 = optimizableVariable('g7',g_range(7,:));
                g8 = optimizableVariable('g8',g_range(8,:));
            end
            n1 = optimizableVariable('n1',n_range);
            n2 = optimizableVariable('n2',n_range);
            n3 = optimizableVariable('n3',n_range);
            n4 = optimizableVariable('n4',n_range);
            n5 = optimizableVariable('n5',n_range);
            n6 = optimizableVariable('n6',n_range);
            n7 = optimizableVariable('n7',n_range);
            n8 = optimizableVariable('n8',n_range);
            vr = optimizableVariable('vr',vr_range);
            
            fun = @(x)cv_bayesopt_fun(X,Y,ID,m_cv_folds,[...
                x.g1 x.n1 x.g2 x.n2 x.g3 x.n3 x.g4 x.n4...
                x.g5 x.n5 x.g6 x.n6 x.g7 x.n7 x.g8 x.n8...
                x.vr],cal_mode,dec_mode);
            results = bayesopt(fun,[...
                g1,n1,g2,n2,g3,n3,g4,n4...
                g5,n5,g6,n6,g7,n7,g8,n8...
                vr],'UseParallel',true,'MaxObjectiveEvaluations',bayes_reps);
            
            % Return best (estimated) hyper-parameters along with their measured error.
            h_params(i,1) = results.XAtMinEstimatedObjective.g1;
            h_params(i,2) = results.XAtMinEstimatedObjective.n1;
            h_params(i,3) = results.XAtMinEstimatedObjective.g2;
            h_params(i,4) = results.XAtMinEstimatedObjective.n2;
            h_params(i,5) = results.XAtMinEstimatedObjective.g3;
            h_params(i,6) = results.XAtMinEstimatedObjective.n3;
            h_params(i,7) = results.XAtMinEstimatedObjective.g4;
            h_params(i,8) = results.XAtMinEstimatedObjective.n4;
            h_params(i,9) = results.XAtMinEstimatedObjective.g5;
            h_params(i,10) = results.XAtMinEstimatedObjective.n5;
            h_params(i,11) = results.XAtMinEstimatedObjective.g6;
            h_params(i,12) = results.XAtMinEstimatedObjective.n6;
            h_params(i,13) = results.XAtMinEstimatedObjective.g7;
            h_params(i,14) = results.XAtMinEstimatedObjective.n7;
            h_params(i,15) = results.XAtMinEstimatedObjective.g8;
            h_params(i,16) = results.XAtMinEstimatedObjective.n8;
            h_params(i,17) = results.XAtMinEstimatedObjective.vr;
            
            
            
            %%%% EXPERIMENTAL %%%%
            
            
            
        end
    end
end