function kruskal_display_figures(Tkruskal,bagProperties,viewMode)
    % This function takes a tensor Tkruskal in its Kruskal form along with
    % the bagProperties struct and the desired view mode number (1,2,3) and
    % displays every figure 'learnt' through the PARAFAC process.
    %
    % Usage:
    %
    % kruskal_display_figures(Tkruskal,bagProperties,viewMode);
    
    % Find figure dimensions.
    figureDims = [bagProperties.patchHeight,bagProperties.patchWidth];
    
    % Display each figure (number of figures = CPD rank).
    for i = 1:size(Tkruskal{2},2)
        if viewMode==1
            figure; imshow(reshape(Tkruskal{2}(:,i),figureDims));
        elseif viewMode==2
            figure; imagesc(reshape(Tkruskal{2}(:,i),figureDims));
        elseif viewMode==3
            figure; imagesc(reshape(Tkruskal{2}(:,i),figureDims)); colormap gray;
        elseif viewMode==4
            buffer = zeros(bagProperties.patchHeight,bagProperties.patchWidth,3);
            buffer(:,:,1) = reshape(Tkruskal{2}(:,i),figureDims) * Tkruskal{3}(1,i);
            buffer(:,:,2) = reshape(Tkruskal{2}(:,i),figureDims) * Tkruskal{3}(2,i);
            buffer(:,:,3) = reshape(Tkruskal{2}(:,i),figureDims) * Tkruskal{3}(3,i);
            figure; imshow(buffer);
        elseif viewMode==5
            imwrite(reshape(Tkruskal{2}(:,i),figureDims),[num2str(i) '.png']);
        elseif viewMode==6
            buffer = zeros(bagProperties.patchHeight,bagProperties.patchWidth,3);
            buffer(:,:,1) = reshape(Tkruskal{2}(:,i),figureDims) * Tkruskal{3}(1,i);
            buffer(:,:,2) = reshape(Tkruskal{2}(:,i),figureDims) * Tkruskal{3}(2,i);
            buffer(:,:,3) = reshape(Tkruskal{2}(:,i),figureDims) * Tkruskal{3}(3,i);
            imwrite(buffer,[num2str(i) '.png']);
        else
            disp('Invalid view mode selected.');
        end
    end
end