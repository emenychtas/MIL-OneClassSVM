function [prcnt] = patch_percent_red(T,bagProperties,patchIndex)
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
    red_pixels = 0;
    for i = 1:nrOfPixels
        if (T(patchIndex,i,1)>T(patchIndex,i,3)); red_pixels = red_pixels + 1; end
    end
    
    % Calculate the mean.
    prcnt = red_pixels / nrOfPixels;
end