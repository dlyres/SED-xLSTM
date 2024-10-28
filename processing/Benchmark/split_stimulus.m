function [] = split_stimulus(stimulus_list, num, fre_points)
if fre_points == 375
    setpath = 'D:\Matlab\workspace\SSVEP\Benchmark\test\SpecialChannelsAndTimes_FIR_DataEnhance_frequence_188\';
else
    setpath = 'D:\Matlab\workspace\SSVEP\Benchmark\test\SpecialChannelsAndTimes_FIR_DataEnhance_frequence_256\';
end

if(num == 8)
    savepath = 'D:\PyCharm 2024.2.0.1\workspace\SSVEP_VIT\dataset\Benchmark\cross_validation\';
else
    savepath = 'D:\PyCharm 2024.2.0.1\workspace\SSVEP_VIT_16\dataset\Benchmark\cross_validation\';
end
class_name = 1;
for element = stimulus_list
    dataset_savepath = strcat(savepath, num2str(class_name), '\');
    mkdir(dataset_savepath);
    dataset_filepath = strcat(setpath, num2str(element), '\');
    data_files = dir(fullfile(dataset_filepath, '*.mat'));
    for i=1:length(data_files)
        filename = data_files(i).name;
        filepath = strcat(dataset_filepath, filename);
        copyfile(filepath, dataset_savepath);
    end
    class_name = class_name + 1;
end

end

