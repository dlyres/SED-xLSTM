function [] = extract_frequence(fre_points)
rootpath = 'D:\Matlab\workspace\SSVEP\BETA\test\';

if fre_points == 1024
    savepath = strcat(rootpath, 'SpecialChannelsAndTimes_FIR_DataEnhance_frequence_512\');
else
    savepath = strcat(rootpath, 'SpecialChannelsAndTimes_FIR_DataEnhance_frequence_256\');
end

mkdir(savepath);

% traversal the folders.
for i=1:40
    
    filepath = strcat(rootpath, 'SpecialChannelsAndTimes_FIR_DataEnhance\', num2str(i), '\');
      
    % Creates a label index folder.
    mkdir(strcat(savepath, num2str(i)));

    % traversal the files
    
    for j=1:880
        setname = strcat(filepath, num2str(j), '_sample.mat');
        EEG = load(setname);

        if fre_points == 1024
            % Number of generated frequency domain information points.
            N = 1024;
            sample_size = size(EEG.sample);
        
            % Generates a three-dimensional array to hold information in the frequency domain.
            sample_frequence = zeros(2, sample_size(1), N/2);
    
            for z=1:sample_size(1)
        
                % Fast Fourier transform.
                fft_data = fft(EEG.sample(z, :),N);
        
                % Amplitude information in the frequency domain.
                fft_data_mod = abs(fft_data(1:N/2));
        
                % Phase information in the frequency domain.
                fft_data_angle = angle(fft_data(1:N/2));
        
                % The cut-off frequency is half of the sampling frequency, and half of the transformed frequency data is retained.
                
                sample_frequence(1, z, :) = fft_data_mod;
                sample_frequence(2, z, :) = fft_data_angle;      
            end
        else
            % Number of generated frequency domain information points.
            N = 512;
            sample_size = size(EEG.sample);
        
            % Generates a three-dimensional array to hold information in the frequency domain.
            sample_frequence = zeros(2, sample_size(1), N/2);
    
            for z=1:sample_size(1)
        
                % Fast Fourier transform.
                fft_data = fft(EEG.sample(z, :), N);
        
                % Amplitude information in the frequency domain.
                fft_data_mod = abs(fft_data(1:N/2));
        
                % Phase information in the frequency domain.
                fft_data_angle = angle(fft_data(1:N/2));
        
                % The cut-off frequency is half of the sampling frequency, and half of the transformed frequency data is retained.
                
                sample_frequence(1, z, :) = fft_data_mod;
                sample_frequence(2, z, :) = fft_data_angle;      
            end
        end
        % Single sample stored separately in a file.
        save_name = strcat(num2str(j), '_sample_fre.mat');
        setpath = strcat(savepath, num2str(i), '\');
        save([setpath, save_name], 'sample_frequence');
    end
 
end
end

