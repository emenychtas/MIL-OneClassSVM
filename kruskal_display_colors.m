buffer = zeros(50,50,3);

for k = 1:size(K,2)
    for i = 1:50
        for j = 1:50
            buffer(i,j,:) = K(:,k);
        end
    end
    
    %figure; imshow(buffer);
    imwrite(buffer,[num2str(k) '.png']);
end