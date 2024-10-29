function [] = data_enhance()

% The time length of 3s is followed by a 1.5-second time window with a step size of 0.5s for sliding segmentation.

rootpath = 'D:\Matlab\workspace\SSVEP\BETA\test\';

folderName = strcat(rootpath,'SpecialChannelsAndTimes_FIR_DataEnhance\');

% Creates a single sample merge folder in the root directory.
mkdir(folderName);

for i=1:40
    savepath = strcat(folderName, num2str(i), '\');

    mkdir(savepath);

    filePath = strcat(rootpath, 'SpecialChannelsAndTimes_FIR\', num2str(i), '\');
    num = 1;
    
    sample = zeros(30, 375);
    for j=1:220
        fileName = strcat(filePath, num2str(j), '_sample.mat');
        EEG = load(fileName);
        starts = 1;
        ends = 375;
        for z=num:num+3
            sample(:, :) = EEG.sample(:, starts:ends);
            save_name = strcat(num2str(z),'_sample.mat');
            save([savepath, save_name],'sample');
            starts = starts + 125;
            ends = ends + 125;   
        end
        num = num + 4;
    end
    
end

disp(num - 1);

end

