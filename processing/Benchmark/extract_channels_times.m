function [result] = extract_channels_times(input_channels,input_times)

rootpath = 'D:\Matlab\workspace\SSVEP\Benchmark\test\';

% Creates a single sample merge folder in the root directory.
mkdir(strcat(rootpath,'SpecialChannelsAndTimes\'))
for i=1:40

    % Creates a label index folder.
    mkdir(strcat(rootpath,'SpecialChannelsAndTimes\', num2str(i)));

    % concatenated file name.
    setname = strcat(rootpath, num2str(i), '_simple_situation.mat');

    % Import data.
    EEG = load(setname);

    % Defines an empty two-dimensional array to store a single sample.
    sample = zeros(length(input_channels), length(input_times));

    % The EEG data of the same type of stimulation were grouped into a single sample.
    for j=1:210
        for k=1:length(input_channels)
            sample(k, :) = EEG.simple_situation(input_channels(k), input_times, j);
        end
        % Single sample stored separately in a file.
        save_name = strcat(num2str(j),'_sample.mat');
        save_path = strcat(rootpath, '\SpecialChannelsAndTimes\', num2str(i), '\');
        save([save_path, save_name],'sample');
    end
end

result = sample;
end

