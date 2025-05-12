%------------------------------------------------
%特定通道特定时间
%0.5s-5.5s
right_sample = 5.5*1500/6;
left_sample = 0.5*1500/6;
times = left_sample+1:right_sample;
length(times)


channels_1 = 54:58;
channels_2 = 61:63;
channels = [48, channels_1, channels_2];
sample = extract_channels_times(channels,times);
size(sample);

%带通滤波范围6-50Hz（包括6Hz、50Hz）
FIR()

%------------------------------------------------
%数据切分增强
time_len = 1.5;
data_enhance(time_len);


%------------------------------------------------
%提取特定通道特定时间频域特征
fre_points = 512;
extract_frequence(fre_points);




%提取十六种刺激到pycharm文件夹中
stimulus_list = [9, 10, 11, 12, 13, 14, 15, 16, 25, 26, 27, 28, 29, 30, 31, 32];
split_stimulus(stimulus_list);



