function patch_display(T,bagProperties,patchIndex)
    % This function takes a multiple-instance image tensor T containing many
    % patches in the INSTANCExPIXELxRGB form, the bagProperties struct and
    % the index of the patch to be displayed and displays it.
    %
    % Usage:
    %
    % patch_display(T,bagProperties,patchIndex);
    
    figure; imshow(reshape(T(patchIndex,:,:),bagProperties.patchHeight,bagProperties.patchWidth,3));
end