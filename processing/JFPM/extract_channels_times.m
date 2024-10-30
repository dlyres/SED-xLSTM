function [result] = extract_channels_times(input_channels,input_times)

rootpath = 'D:\Matlab\workspace\SSVEP\JFPM\test\';

mkdir(strcat(rootpath,'SpecialChannelsAndTimes\'))
for i=1:12

    mkdir(strcat(rootpath,'SpecialChannelsAndTimes\', num2str(i)));

    setname = strcat(rootpath, num2str(i), '_simple_situation.mat');

    EEG = load(setname);

    sample = zeros(length(input_channels), length(input_times));

    for j=1:150
        for k=1:length(input_channels)
            sample(k, :) = EEG.simple_situation(input_channels(k), input_times, j);
        end

        save_name = strcat(num2str(j),'_sample.mat');
        save_path = strcat(rootpath, '\SpecialChannelsAndTimes\', num2str(i), '\');
        save([save_path, save_name],'sample');
    end
end

result = sample;
end

