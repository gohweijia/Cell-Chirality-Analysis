function x=analyse_folders(rootdir)
	log_path = fullfile(rootdir, "matlab_log.txt");
    if exist(log_path)
        delete log_path;
    end
    fileID = fopen(log_path, 'a');
    pixel_size = 0.138502;            % micrometer
    min_length_um = 1;
    nb_thresh = -0.2;
    for i=0:2:14
    	%disp(sprintf('%d - %d', i, i+2));
        analyse_single(rootdir, i, i+4, nb_thresh, min_length_um, pixel_size);
        fprintf(fileID, '%d-%d completed\n', i, i+2);
    end 
    fclose(fileID);
end
