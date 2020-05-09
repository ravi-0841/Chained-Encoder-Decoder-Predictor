function generate_momenta(data_path, fraction)
%     data_path = '/home/ravi/Desktop';
%     fraction = 'test'; % Testing
	try
        rmdir ~/.matlab/local_cluster_jobs;
	catch
		disp("Local cluster jobs directory does not exist");
	end
	
    addpath(genpath('./'));

    parpool(4)
	
    kx = 6;
	ky = 50;
	disp(['kernel x is ', num2str(kx)]);

    data = load(fullfile(data_path, [fraction, '.mat']));
    
    src_f0_feat = double(data.src_f0_feat);
    tar_f0_feat = double(data.tar_f0_feat);

    src_mfc_feat= double(data.src_mfc_feat);
    tar_mfc_feat= double(data.tar_mfc_feat);
    
    dim_f0 		= size(src_f0_feat,3);
    x 			= 1:dim_f0;
    momenta_f0  = zeros(size(src_f0_feat));
    N_samples   = size(src_f0_feat,2);
    
    for file_idx = 1:size(src_f0_feat,1)
        parfor sample = 1:N_samples
            y_s = squeeze(src_f0_feat(file_idx, sample, :));
            y_t = squeeze(tar_f0_feat(file_idx, sample, :));
            y_s = [x' y_s];
            y_t = [x' y_t];

            if sum(y_s(:)==0)==length(y_s) || sum(y_t(:)==0)==length(y_t) 
                momenta_f0(file_idx, sample,:) = nan(1,dim_f0);
            else
                try
                    momenta_f0(file_idx, sample,:) = get_momentum(y_s, y_t, kx, ky)';
                catch
                    momenta_f0(file_idx, sample,:) = nan(1,dim_f0);
                end
            end
        end
        disp([num2str(file_idx), ' Processed']);
    end
        
    save(fullfile(data_path, ['momenta_', fraction, '.mat']), 'src_mfc_feat', 'src_f0_feat',...
                                                                'momenta_f0', 'tar_mfc_feat', ...
                                                                'tar_f0_feat');

