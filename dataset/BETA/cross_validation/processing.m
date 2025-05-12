%------------------------------------------------
%特定通道特定时间
%0.5s-3.5s
right_sample = 3.5*1000/4;
left_sample = 0.5*1000/4;
times = left_sample+1:right_sample;
length(times)

%索引后脑顶叶区域和枕叶区域附近的共9个导联电极
channels_1 = 54:58;
channels_2 = 61:63;
channels = [48, channels_1, channels_2];
sample = extract_channels_times(channels,times);
size(sample);

%带通滤波范围6-50Hz（包括6Hz、50Hz）
FIR();

%------------------------------------------------
%数据切分增强
data_enhance();


%------------------------------------------------
%提取特定通道特定时间频域特征
fre_points = 512;
extract_frequence(fre_points);

%------------------------------------------------
%时域频域融合
time_fre();

%划分数据集
split_train_test()


