function [svm_model] = suppressed_svmtrain(Y,X,svm_options)
    % This is a function wrapper for svmtrain which suppresses the output.
    % It uses the evalc function to capture the output and then ignores it.
    %
    % Usage:
    %
    % [svm_model] = suppressed_svmtrain(Y,X,svm_options);

    [~,svm_model] = evalc("svmtrain(Y,X,svm_options)");
end