function generate_vertical_data(file_idx, emo, mode)
	try
        rmdir ~/.matlab/local_cluster_jobs;
	catch
		disp("Local cluster jobs directory does not exist");
	end
	addpath(genpath('./'));
	% parpool(4);
    file_idx = str2num(file_idx);
	kx = 6;
	ky = 50;
	disp(['kernel x is ', num2str(kx)]);

	if strcmp(mode, 'train')

		data = load(['./data/', emo, '/train.mat']);
        
        src_f0_feat = double(data.src_f0_feat);
        tar_f0_feat = double(data.tar_f0_feat);

        src_ec_feat = double(data.src_ec_feat);
        tar_ec_feat = double(data.tar_ec_feat);

        src_log_f0_feat = double(data.src_log_f0_feat);
        tar_log_f0_feat = double(data.tar_log_f0_feat);

        src_f0_feat = squeeze(src_f0_feat(file_idx,:,:));
        tar_f0_feat = squeeze(tar_f0_feat(file_idx,:,:));
        src_ec_feat = squeeze(src_ec_feat(file_idx,:,:));
        tar_ec_feat = squeeze(tar_ec_feat(file_idx,:,:));
        src_log_f0_feat = squeeze(src_log_f0_feat(file_idx,:,:));
        tar_log_f0_feat = squeeze(tar_log_f0_feat(file_idx,:,:));
        
        x 			= 1:size(src_f0_feat,2);
        dim_f0 		= size(src_f0_feat,2);
        dim_log_f0  = size(src_log_f0_feat,2);
        dim_ec 		= size(src_ec_feat,2);

		momentum_f0       = zeros(size(src_f0_feat,1), size(src_f0_feat,2));
		momentum_log_f0   = zeros(size(src_log_f0_feat,1), size(src_log_f0_feat,2));
		momentum_ec       = zeros(size(src_f0_feat,1), size(src_ec_feat,2));
        
		for sample = 1:size(src_f0_feat,1)
		    y_s = src_f0_feat(sample,:);
		    y_t = tar_f0_feat(sample,:);
            y_s = [x' y_s'];
            y_t = [x' y_t'];

            v_s = src_log_f0_feat(sample,:);
		    v_t = tar_log_f0_feat(sample,:);
            v_s = [x' v_s'];
            v_t = [x' v_t'];

            z_s = src_ec_feat(sample,:);
		    z_t = tar_ec_feat(sample,:);
            z_s = [x' z_s'];
            z_t = [x' z_t']; 

		    if sum(y_s(:)==0)==length(y_s) || sum(y_t(:)==0)==length(y_t) 
		    	momentum_f0(sample,:) = nan(1,dim_f0);
		   	else
		        try
		            momentum_f0(sample,:) = get_momentum(y_s, y_t, kx, 50)';
                catch
		            momentum_f0(sample,:) = nan(1,dim_f0);
		        end
            end

		   	if sum(v_s(:)==0)==length(v_s) || sum(v_t(:)==0)==length(v_t) 
		    	momentum_log_f0(sample,:) = nan(1,dim_log_f0);
            else
		        try
		            momentum_log_f0(sample,:) = get_momentum(v_s, v_t, kx, 0.5)';
                catch
		            momentum_log_f0(sample,:) = nan(1,dim_log_f0);
		        end
            end

		   	if sum(z_s(:)==0)==length(z_s) || sum(z_t(:)==0)==length(z_t) 
		    	momentum_ec(sample,:) = nan(1,dim_ec);
            else
		        try
		            momentum_ec(sample,:) = get_momentum(z_s, z_t, kx, 1.0)';
                catch
		            momentum_ec(sample,:) = nan(1,dim_ec);
		        end
            end
		    disp([num2str(sample), ' Processed']);
		end
		size(momentum_f0)
        size(momentum_ec)
        size(momentum_log_f0)
		save(['./data/',emo,'/f0-train-', num2str(file_idx), '.mat'], 'momentum_f0');
		save(['./data/',emo,'/log-f0-train-', num2str(file_idx), '.mat'], 'momentum_log_f0');
		save(['./data/',emo,'/ec-train-', num2str(file_idx), '.mat'], 'momentum_ec');

	elseif strcmp(mode, 'valid')	
		data = load(['./data/', emo, '/valid.mat']);
        
        src_f0_feat = double(data.src_f0_feat);
        tar_f0_feat = double(data.tar_f0_feat);

        src_ec_feat = double(data.src_ec_feat);
        tar_ec_feat = double(data.tar_ec_feat);

        src_log_f0_feat = double(data.src_log_f0_feat);
        tar_log_f0_feat = double(data.tar_log_f0_feat);

        src_f0_feat = squeeze(src_f0_feat(file_idx,:,:));
        tar_f0_feat = squeeze(tar_f0_feat(file_idx,:,:));
        src_ec_feat = squeeze(src_ec_feat(file_idx,:,:));
        tar_ec_feat = squeeze(tar_ec_feat(file_idx,:,:));
        src_log_f0_feat = squeeze(src_log_f0_feat(file_idx,:,:));
        tar_log_f0_feat = squeeze(tar_log_f0_feat(file_idx,:,:));
        
        x 			= 1:size(src_f0_feat,2);
        dim_f0 		= size(src_f0_feat,2);
        dim_log_f0  = size(src_log_f0_feat,2);
        dim_ec 		= size(src_ec_feat,2);

		momentum_f0       = zeros(size(src_f0_feat,1), size(src_f0_feat,2));
		momentum_log_f0   = zeros(size(src_log_f0_feat,1), size(src_log_f0_feat,2));
		momentum_ec       = zeros(size(src_f0_feat,1), size(src_ec_feat,2));
        
		for sample = 1:size(src_f0_feat,1)
		    y_s = src_f0_feat(sample,:);
		    y_t = tar_f0_feat(sample,:);
            y_s = [x' y_s'];
            y_t = [x' y_t'];

            v_s = src_log_f0_feat(sample,:);
		    v_t = tar_log_f0_feat(sample,:);
            v_s = [x' v_s'];
            v_t = [x' v_t'];

            z_s = src_ec_feat(sample,:);
		    z_t = tar_ec_feat(sample,:);
            z_s = [x' z_s'];
            z_t = [x' z_t']; 

		    if sum(y_s(:)==0)==length(y_s) || sum(y_t(:)==0)==length(y_t) 
		    	momentum_f0(sample,:) = nan(1,dim_f0);
		   	else
		        try
		            momentum_f0(sample,:) = get_momentum(y_s, y_t, kx, 50)';
                catch
		            momentum_f0(sample,:) = nan(1,dim_f0);
		        end
            end

		   	if sum(v_s(:)==0)==length(v_s) || sum(v_t(:)==0)==length(v_t) 
		    	momentum_log_f0(sample,:) = nan(1,dim_log_f0);
            else
		        try
		            momentum_log_f0(sample,:) = get_momentum(v_s, v_t, kx, 0.5)';
                catch
		            momentum_log_f0(sample,:) = nan(1,dim_log_f0);
		        end
            end

		   	if sum(z_s(:)==0)==length(z_s) || sum(z_t(:)==0)==length(z_t) 
		    	momentum_ec(sample,:) = nan(1,dim_ec);
            else
		        try
		            momentum_ec(sample,:) = get_momentum(z_s, z_t, kx, 1.0)';
                catch
		            momentum_ec(sample,:) = nan(1,dim_ec);
		        end
            end
		    disp([num2str(sample), ' Processed']);
		end
		size(momentum_f0)
        size(momentum_ec)
        size(momentum_log_f0)
		save(['./data/',emo,'/f0-valid-', num2str(file_idx), '.mat'], 'momentum_f0');
		save(['./data/',emo,'/log-f0-valid-', num2str(file_idx), '.mat'], 'momentum_log_f0');
		save(['./data/',emo,'/ec-valid-', num2str(file_idx), '.mat'], 'momentum_ec');

	elseif strcmp(mode, 'test')
		data = load(['./data/', emo, '/test.mat']);
        
        src_f0_feat = double(data.src_f0_feat);
        tar_f0_feat = double(data.tar_f0_feat);

        src_ec_feat = double(data.src_ec_feat);
        tar_ec_feat = double(data.tar_ec_feat);

        src_log_f0_feat = double(data.src_log_f0_feat);
        tar_log_f0_feat = double(data.tar_log_f0_feat);

        src_f0_feat = squeeze(src_f0_feat(file_idx,:,:));
        tar_f0_feat = squeeze(tar_f0_feat(file_idx,:,:));
        src_ec_feat = squeeze(src_ec_feat(file_idx,:,:));
        tar_ec_feat = squeeze(tar_ec_feat(file_idx,:,:));
        src_log_f0_feat = squeeze(src_log_f0_feat(file_idx,:,:));
        tar_log_f0_feat = squeeze(tar_log_f0_feat(file_idx,:,:));
        
        x 			= 1:size(src_f0_feat,2);
        dim_f0 		= size(src_f0_feat,2);
        dim_log_f0  = size(src_log_f0_feat,2);
        dim_ec 		= size(src_ec_feat,2);

		momentum_f0       = zeros(size(src_f0_feat,1), size(src_f0_feat,2));
		momentum_log_f0   = zeros(size(src_log_f0_feat,1), size(src_log_f0_feat,2));
		momentum_ec       = zeros(size(src_f0_feat,1), size(src_ec_feat,2));
        
		for sample = 1:size(src_f0_feat,1)
		    y_s = src_f0_feat(sample,:);
		    y_t = tar_f0_feat(sample,:);
            y_s = [x' y_s'];
            y_t = [x' y_t'];

            v_s = src_log_f0_feat(sample,:);
		    v_t = tar_log_f0_feat(sample,:);
            v_s = [x' v_s'];
            v_t = [x' v_t'];

            z_s = src_ec_feat(sample,:);
		    z_t = tar_ec_feat(sample,:);
            z_s = [x' z_s'];
            z_t = [x' z_t']; 

		    if sum(y_s(:)==0)==length(y_s) || sum(y_t(:)==0)==length(y_t) 
		    	momentum_f0(sample,:) = nan(1,dim_f0);
		   	else
		        try
		            momentum_f0(sample,:) = get_momentum(y_s, y_t, kx, 50)';
                catch
		            momentum_f0(sample,:) = nan(1,dim_f0);
		        end
            end

		   	if sum(v_s(:)==0)==length(v_s) || sum(v_t(:)==0)==length(v_t) 
		    	momentum_log_f0(sample,:) = nan(1,dim_log_f0);
            else
		        try
		            momentum_log_f0(sample,:) = get_momentum(v_s, v_t, kx, 0.5)';
                catch
		            momentum_log_f0(sample,:) = nan(1,dim_log_f0);
		        end
            end

		   	if sum(z_s(:)==0)==length(z_s) || sum(z_t(:)==0)==length(z_t) 
		    	momentum_ec(sample,:) = nan(1,dim_ec);
            else
		        try
		            momentum_ec(sample,:) = get_momentum(z_s, z_t, kx, 1.0)';
                catch
		            momentum_ec(sample,:) = nan(1,dim_ec);
		        end
            end
		    disp([num2str(sample), ' Processed']);
		end
		size(momentum_f0)
        size(momentum_ec)
        size(momentum_log_f0)
		save(['./data/',emo,'/f0-test-', num2str(file_idx), '.mat'], 'momentum_f0');
		save(['./data/',emo,'/log-f0-test-', num2str(file_idx), '.mat'], 'momentum_log_f0');
		save(['./data/',emo,'/ec-test-', num2str(file_idx), '.mat'], 'momentum_ec');
	end
	exit;

%clear num_samples sample mom src_contour tar_contour
