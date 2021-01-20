function [Xpca] = matrix_pca_variance_retained(X,variance_retained)
    % This function takes a matrix X along with a percentage value within
    % the range of [0,1]. It performs PCA on the matrix X and returns the
    % necessary amount of columns (starting from the first) of the scores
    % matrix in order to have the desired variance retained.
    %
    % Usage:
    %
    % [Xpca] = matrix_pca_variance_retained(X,variance_retained);
    
    % Perform PCA on matrix X.
    [~,score,~,~,explained,~] = pca(X);
    
    % Find out how many principal components to keep.
    variance_explained = 0;
    for i = 1:size(X,2)
        variance_explained = variance_explained + explained(i);
        if (variance_explained >= (variance_retained * 100))
            break;
        end
    end
    
    % Return only needed principal components.
    Xpca = score(:,1:i);
end