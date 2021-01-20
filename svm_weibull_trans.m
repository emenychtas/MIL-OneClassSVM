function [norm_dec_values] = svm_weibull_trans(dec_values,W,apriori,mode)
    % Pre-allocate new decision values.
    norm_dec_values = zeros(size(dec_values));
    
    % For each SVM.
    for i = 1:size(dec_values,2)
        % For each instance.
        for j = 1:size(dec_values,1)
            if isequal(mode,'open')
                % OPEN SET - UN-NORMALIZED BAYES
                norm_dec_values(j,i) = wblcdf(dec_values(j,i),W(i,1),W(i,2)) * apriori(i);
            elseif isequal(mode,'closed')
                % CLOSED SET - FULL BAYES THEOREM
                divider = 0;
                % For each SVM.
                for k = 1:size(dec_values,2)
                    divider = divider + wblcdf(dec_values(j,i),W(k,1),W(k,2)) * apriori(k);
                end
                norm_dec_values(j,i) = wblcdf(dec_values(j,i),W(i,1),W(i,2)) * apriori(i) / divider;
                if isnan(norm_dec_values(j,i))
                    norm_dec_values(j,i) = 0;
                end
            end
        end
    end
end