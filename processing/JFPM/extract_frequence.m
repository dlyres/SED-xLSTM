function [] = extract_frequence(fre_points)

rootpath = 'D:\Matlab\workspace\SSVEP\JFPM\test\';
if fre_points == 384
    savepath = strcat(rootpath, 'SpecialChannelsAndTimes_FIR_DataEnhance_frequence_192\');
else
    savepath = strcat(rootpath, 'SpecialChannelsAndTimes_FIR_DataEnhance_frequence_256\');
end
mkdir(savepath);

for i=1:12
    
    filepath = strcat(rootpath, 'SpecialChannelsAndTimes_FIR_DataEnhance\', num2str(i), '\');
      
    mkdir(strcat(savepath, num2str(i)));
    
    for j=1:900
        setname = strcat(filepath, num2str(j), '_sample.mat');
        EEG = load(setname);

        if fre_points == 384
            N = 384;
        else
            N = 512;
        end
        sample_size = size(EEG.sample);
    
        sample_frequence = zeros(2, sample_size(1), N/2);

        for z=1:sample_size(1)
    
            fft_data = fft(EEG.sample(z, :),N);

            fft_data_mod = abs(fft_data(1:N/2));

            fft_data_angle = angle(fft_data(1:N/2));
            
            sample_frequence(1, z, :) = fft_data_mod;
            sample_frequence(2, z, :) = fft_data_angle;      
        end
    
        save_name = strcat(num2str(j), '_sample_fre.mat');
        setpath = strcat(savepath, num2str(i), '\');
        save([setpath, save_name], 'sample_frequence');
    end
 
end
end

