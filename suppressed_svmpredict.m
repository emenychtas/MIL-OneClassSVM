function [pred_labels,accuracy,dec_values] = suppressed_svmpredict(Y,X,svm_model)
    % This is a function wrapper for svmpredict which suppresses the output.
    % It uses the evalc function to capture the output and then ignores it.
    %
    % Usage:
    %
    % [pred_labels,accuracy,dec_values] = suppressed_svmpredict(Y,X,svm_model);
    
    [~,pred_labels,accuracy,dec_values] = evalc("svmpredict(Y,X,svm_model)");
end