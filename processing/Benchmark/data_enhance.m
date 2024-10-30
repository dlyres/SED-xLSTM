function [] = data_enhance(time_len)

rootpath = 'D:\Matlab\workspace\SSVEP\Benchmark\test\';

% Move 3s to a 1.5-second time window with a step size of 0.5s for sliding segmentation.

if time_len == 1.5
    folderName = strcat(rootpath,'SpecialChannelsAndTimes_FIR_DataEnhance\');
    
    % Create a single sample merge folder in the root directory.
    mkdir(folderName);
    
    for i=1:40
        savepath = strcat(folderName, num2str(i), '\');
    
        mkdir(savepath);
    
        filePath = strcat(rootpath, 'SpecialChannelsAndTimes_FIR\', num2str(i), '\');
        num = 1;
        
        sample = zeros(30, 375);
        for j=1:210
            fileName = strcat(filePath, num2str(j), '_sample.mat');
            EEG = load(fileName);
            starts = 1;
            ends = 375;
            for z=num:num+7
                sample(:, :) = EEG.sample(:, starts:ends);
                save_name = strcat(num2str(z),'_sample.mat');
                save([savepath, save_name],'sample');
                starts = starts + 125;
                ends = ends + 125;   
            end
            num = num + 8;
        end
        
    end
    
    disp(num - 1);
end
end

