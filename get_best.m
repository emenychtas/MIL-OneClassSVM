function [metrics] = get_best(metric,method,results)
    reps = size(results,1) / 12;
    ind_start = (method - 1) * reps + 1;
    ind_end = method * reps;
    temp = cell2mat(results(ind_start:ind_end,[1,2,3,4]));
    
    if isequal(metric,'acc')
        [~,ind] = max(temp(:,1)-temp(:,2));
    elseif isequal(metric,'auc')
        [~,ind] = max(temp(:,3)-temp(:,4));
    end
    
    metrics = temp(ind,:);
end