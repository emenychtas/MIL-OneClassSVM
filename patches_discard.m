function [new_T,new_Y,new_ID] = patches_discard(T,Y,ID,bagProperties,discRule)
    % This function takes a multiple-instance image tensor T containing many
    % patches in the INSTANCExPIXELxRGB form, the Y matrix containing the
    % label of each patch, the ID matrix connecting every patch to its bag,
    % the bagProperties struct and a discard rule number.
    % It discards patches along with the corresponding entries in the other
    % matrices (Y and ID) according to the specified rule.
    %
    % Usage:
    %
    % [new_T,new_Y,new_ID] = patches_discard(T,Y,ID,bagProperties,discRule);
    
    % Check if the specified discard rule exists.
    if ~ismember(discRule,[1])
        disp('Invalid discard rule number.');
    else
        nrOfSubjects = length(unique(ID));
        nrOfPatchesPerSubj = size(T,1)/nrOfSubjects;
        
        % Indices of the remaining patches.
        remaining_indices = nan(nrOfSubjects,nrOfPatchesPerSubj);
        discarded_indices = nan(nrOfSubjects,nrOfPatchesPerSubj);
        
        % For every subject.
        for j = 1:nrOfSubjects
            patches_kept = 0;
            threshold = 0.5;
            
            percent_white = nan(nrOfPatchesPerSubj);
            percent_red = nan(nrOfPatchesPerSubj);
            
            parfor i = 1:nrOfPatchesPerSubj
                percent_white(i) = patch_percent_white(T,bagProperties,i+(j-1)*nrOfPatchesPerSubj);
                percent_red(i) = patch_percent_red(T,bagProperties,i+(j-1)*nrOfPatchesPerSubj);
            end
            
            while patches_kept<50
                % Start with a clean count.
                patches_kept = 0;
                
                % Check if every single patch should be kept or not.
                for i = 1:nrOfPatchesPerSubj
                    patch_keep = 1;
                    
                    % Pick a single rule.
                    if discRule==1
                        if (percent_white(i) + percent_red(i)) > threshold; patch_keep = 0; end
                    end
                    
                    % Add patch to the list of patches to keep or display the discarded patch.
                    if patch_keep==1
                        remaining_indices(j,i) = i+(j-1)*nrOfPatchesPerSubj;
                        patches_kept = patches_kept + 1;
                    else
                        discarded_indices(j,i) = i+(j-1)*nrOfPatchesPerSubj;
                    end
                end
                
                % Increase threshold for next iteration if needed.
                threshold = threshold + 0.01;
            end
        end
        
        remaining_indices = remaining_indices(:);
        discarded_indices = discarded_indices(:);
        remaining_indices = remaining_indices(~isnan(remaining_indices));
        discarded_indices = discarded_indices(~isnan(discarded_indices));
        
        % Pre-allocate and fill new tensor/matrices.
        new_T = uint8(-ones(length(remaining_indices),size(T,2),3));
        new_Y = -ones(length(remaining_indices),1);
        new_ID = -ones(length(remaining_indices),1);
        parfor i = 1:length(remaining_indices)
            new_T(i,:,:) = T(remaining_indices(i),:,:);
            new_Y(i) = Y(remaining_indices(i));
            new_ID(i) = ID(remaining_indices(i));
        end
        
        % Display kept patches.
        patches_to_bag(new_T,new_ID,bagProperties,1);
        
        % Pre-allocate and fill tensor/matrices for discarded patches.
        disc_T = uint8(-ones(length(discarded_indices),size(T,2),3));
        disc_Y = -ones(length(discarded_indices),1);
        disc_ID = -ones(length(discarded_indices),1);
        parfor i = 1:length(discarded_indices)            
            disc_T(i,:,:) = T(discarded_indices(i),:,:);
            disc_Y(i) = Y(discarded_indices(i));
            disc_ID(i) = ID(discarded_indices(i));
        end
        
        % Display discarded patches.
        patches_to_bag(disc_T,disc_ID,bagProperties,1);
    end
end