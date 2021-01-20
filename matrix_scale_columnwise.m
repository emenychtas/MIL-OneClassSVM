function [Xscl] = matrix_scale_columnwise(X,bounds)
    % This function takes a matrix X along with a vector containing the
    % bounds (this can be [0,1] or [-1,1]). It scales each column of the
    % matrix X independently withing the given bounds and returns it.
    %
    % Usage:
    %
    % [Xscl] = matrix_scale_columnwise(X,bounds);
    
    if isequal(bounds,[0,1])
        Xmin = min(X);
        Xmax = max(X);
        Xscl = (X - Xmin) ./ (Xmax - Xmin + eps);
    elseif isequal(bounds,[-1,1])
        Xmax = max(abs(X));
        Xscl = X ./ (Xmax + eps);
    elseif isequal(bounds,'soft')
        Xmean = repmat(mean(X),size(X,1),1);
        Xstd = std(X);
        Xscl = (X - Xmean) ./ (2 * Xstd + eps);
    elseif isequal(bounds,'zscore')
        Xscl = zscore(X);
    else
        disp('Invalid scaling bounds. Only [0,1] and [-1,1] are supported.');
    end
end