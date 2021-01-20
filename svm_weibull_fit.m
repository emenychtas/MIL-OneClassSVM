function [W] = svm_weibull_fit(dec_values,svm_models,tail_size)
    % Pre-allocate Weibull PDF parameter matrix.
    W = zeros(length(dec_values),2);
    
    % For each SVM, fit a Weibull PDF.
    for i = 1:length(dec_values)
        % Get SVM's training decision values and keep only the positive
        % ones in ascending order.
        V = dec_values{i};
        V = V(V>0);
        V = sort(V,'ascend');
        
        % Calculate how many of them to use for the fit, starting from the
        % smallest one.
        if tail_size == -1
            % 1.5 Times POSITIVE SVs
            [~,~,sv_dvals] = suppressed_svmpredict(ones(svm_models{i}.totalSV,1),svm_models{i}.SVs,svm_models{i});
            n = ceil(1.5 * nnz(sv_dvals > 0));
            
            % 1.5 Times ALL SVs
            %n = ceil(1.5 * svm_models{i}.totalSV);
        else
            n = ceil(tail_size * length(V));
        end
        n = max([3,n]);         % Minimum 3 smallest positive dec values.
        n = min([n,length(V)]); % Maximum all of the positive dec values.
        
        % Fit the PDF and store the parameters.
        D = V(1:n);
        %figure; plot(D);
        W(i,:) = wblfit(D);
    end
end