% Extract he same stimulus type from all subjects.
save_path = 'D:\Matlab\workspace\SSVEP\BETA\test\';
setpath = 'D:\Matlab\workspace\SSVEP\BETA\dataset\';
for j = 1:40

    % Create an empty 3D array that stores the same stimulus type.
    simple_situation = zeros(64, 1000, 4*55);
    start = 1;
    step = 3;
    for i = 16:70

        % Concat files name.
        setname = strcat(setpath, 'S', num2str(i), '.mat');

        % import data.
        data_ori = load(setname);

        % Summary of the same stimulus type for each subject.
        simple_situation(:, :, start:start + step) = data_ori.data.EEG(:, :, :, j);
        start = start + step + 1;
    end

    % save files.
    save_name = strcat(num2str(j),'_simple_situation.mat');
    save([save_path, save_name],'simple_situation');
end

%------------------------------------------------
% Whole channels and whole time length.
% create label index.
rootpath = 'D:\Matlab\workspace\SSVEP\BETA\test\';
for i=1:40

    % create label index folders.
    mkdir(strcat(rootpath, 'AllchannelsAndTimes\', num2str(i)));

    % concat file name.
    setname = strcat(rootpath, num2str(i), '_simple_situation.mat');

    % import data.
    EEG = load(setname);

    % Define an empty two-dimensional array to store a single sample.
    sample = zeros(64, 1000);

    % The EEG data of the same stimulation type were composed into a single sample.
    for j=1:220
        sample(:, :) = EEG.simple_situation(:, :, j);

        % A single sample is stored in a single file.
        save_name = strcat(num2str(j),'_sample.mat');
        save_path = strcat(rootpath, 'AllchannelsAndTimes\', num2str(i), '\');
        save([save_path, save_name],'sample');
    end
end

%------------------------------------------------
% Specific channel, specific time.
%0.5s-3.5s
right_sample = 3.5*1000/4;
left_sample = 0.5*1000/4;
times = left_sample+1:right_sample;
length(times)

% A total of 30 lead electrodes near the parietal and occipital regions of the posterior brain were indexed.
channels_1 = 34:42;
channels_2 = 44:64;
channels = [channels_1, channels_2];
sample = extract_channels_times(channels,times);
size(sample);

% Band-pass filtering range 6-50Hz（includes 6Hz、50Hz）.
FIR()

%------------------------------------------------
% data clip enhance.
data_enhance()


%------------------------------------------------
% Extract specific time-frequency domain features of specific channels.
fre_points = 1024;
extract_frequence(fre_points);


% Extract the eight stimuli into the pycharm folder.
num = 8;
fre_points = 512;
stimulus_list = [1, 6, 11, 16, 21, 26, 31, 36];
split_stimulus(stimulus_list, num, fre_points);





