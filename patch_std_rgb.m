function [std_rgb] = patch_std_rgb(T,bagProperties,patchIndex)
    % This function takes a multiple-instance image tensor T containing many
    % patches in the INSTANCExPIXELxRGB form, the bagProperties struct and
    % the index of the patch to be measured and returns its mean RGB values.
    %
    % Usage:
    %
    % [mean_rgb] = patch_mean_rgb(T,bagProperties,patchIndex);
    
    % Calculate number of pixels.
    nrOfPixels = bagProperties.patchWidth*bagProperties.patchHeight;
    
    % Sum the colors of all pixels per channel.
    rgb_sums = -ones(nrOfPixels,1);
    for i = 1:nrOfPixels
        rgb_sums(i) = sum(T(patchIndex,i,:));
    end
    
    % Calculate the mean.
    std_rgb = std(rgb_sums);
end