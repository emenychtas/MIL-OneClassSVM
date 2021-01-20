function [mean_rgb] = patch_mean_rgb(T,bagProperties,patchIndex)
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
    mean_rgb = [0,0,0];
    for i = 1:nrOfPixels
        mean_rgb(1) = mean_rgb(1) + double(T(patchIndex,i,1));
        mean_rgb(2) = mean_rgb(2) + double(T(patchIndex,i,2));
        mean_rgb(3) = mean_rgb(3) + double(T(patchIndex,i,3));
    end
    
    % Calculate the mean.
    mean_rgb = mean_rgb / nrOfPixels;
end