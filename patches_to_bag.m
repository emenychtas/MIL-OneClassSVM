function [bag] = patches_to_bag(T,ID,bagProperties,bagID)
    % This function takes a multiple-instance image tensor T containing many
    % patches in the INSTANCExPIXELxRGB form, the ID matrix connecting every
    % patch to its bag, the bagProperties struct and the ID of the bag to be
    % reconstructed.
    % In case the patches have been produced with an overlap factor higher
    % than 1, or if there've been patches discarded, or both, this function
    % will still work, but the result will most likely have some zero padding.
    %
    % Usage:
    %
    % [bag] = patches_to_bag(T,ID,bagProperties,bagID);
    
    % Keep only the patches that belong to the specified bag.
    T = T(ID==bagID,:,:);
    
    % Add every patch to the bag (without pre-allocation).
    bag = uint8([]);
    for i = 0:size(T,1)-1
        pad_x = floor(i/bagProperties.gridResolution) * bagProperties.patchHeight;
        pad_y = mod(i,bagProperties.gridResolution) * bagProperties.patchWidth;
        
        bag((1+pad_x):(bagProperties.patchHeight+pad_x),(1+pad_y):(bagProperties.patchWidth+pad_y),:) = ...
            reshape(T(i+1,:,:),bagProperties.patchHeight,bagProperties.patchWidth,3);
            
    end
    
    % Display the final bag.
    if ~isempty(bag)
        figure('Name',['Subject #' num2str(bagID) ' Informative Windows']); imshow(bag);
    else
        disp('Bag is empty.');
    end
end