function [prcnt] = patch_percent_white(T,bagProperties,patchIndex)
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
    white_pixels = 0;
    for i = 1:nrOfPixels
        if (sum(T(patchIndex,i,:))>255*3*0.85); white_pixels = white_pixels + 1; end
    end
    
    % Calculate the mean.
    prcnt = white_pixels / nrOfPixels;
end