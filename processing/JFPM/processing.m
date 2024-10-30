save_path = 'D:\Matlab\workspace\SSVEP\JFPM\test\';
setpath = 'D:\Matlab\workspace\SSVEP\JFPM\dataset\';
for j = 1:12

    simple_situation = zeros(8, 1114, 15*10);
    start = 1;
    step = 14;
    for i = 1:10

        setname = strcat(setpath, 'S', num2str(i), '.mat');

        data = load(setname);

        data_ori = permute(data.eeg, [2, 3, 4, 1]);

        simple_situation(:, :, start:start + step) = data_ori(:, :, :, j);
        start = start + step + 1;
    end

    save_name = strcat(num2str(j),'_simple_situation.mat');
    save([save_path, save_name],'simple_situation');
end

%------------------------------------------------
% Whole channels and whole time length.
% create label index.
rootpath = 'D:\Matlab\workspace\SSVEP\JFPM\test\';
for i=1:12

    mkdir(strcat(rootpath, 'AllchannelsAndTimes\', num2str(i)));

    setname = strcat(rootpath, num2str(i), '_simple_situation.mat');

    EEG = load(setname);
    
    sample = zeros(8, 1114);

    for j=1:150
        sample(:, :) = EEG.simple_situation(:, :, j);
        save_name = strcat(num2str(j),'_sample.mat');
        save_path = strcat(rootpath, 'AllchannelsAndTimes\', num2str(i), '\');
        save([save_path, save_name],'sample');
    end
end

%------------------------------------------------
% Specific channel, specific time.
%0.15s-4.15s
right_sample = 1063;
left_sample = 39;
input_times = left_sample+1:right_sample;
length(input_times)

% Index all 8 lead electrodes.
input_channels = [1, 2, 3, 4, 5, 6, 7, 8];
sample = extract_channels_times(input_channels, input_times);
size(sample);

FIR()

%------------------------------------------------
data_enhance()


%------------------------------------------------
fre_points = 512;
extract_frequence(fre_points);


% Extract all stimuli into the pycharm folder.
stimulus_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
fre_points = 512;
split_stimulus(stimulus_list, fre_points);







