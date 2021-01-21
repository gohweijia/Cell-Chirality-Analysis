skip_completed = true;  % Change this to 'false' to re-run analysis for complete data
ncpus = feature('numcores')
txt_path = 'paths.txt';
paths = textread(txt_path,'%s','delimiter','\n');
paths(ismember(paths,'')) = [];  
paths(contains(paths,'%')) = []; 

global nb_thresh;
global min_length_um;
global pixel_size;
pixel_size = 0.138502;          
min_length_um = 1;
nb_thresh = -0.2;

job_list = {};
for path_index = 1:length(paths)
    rootdir = paths{path_index};
    disp(rootdir);
    directories = dir(rootdir);
    directories = directories(~contains({directories.name}, '.'));
    for file_index = 1:length(directories)
        pathname = fullfile(rootdir, directories(file_index).name)+ "/";
        if ~exist(fullfile(pathname, 'actin', '14_18', 'Results.csv'))
            job_list{end+1} = pathname;
        end
    end
end

celldisp(job_list);
cluster1 = parcluster;
for job_index = 1:length(job_list)
    job_path = job_list{job_index};
    try
        job(job_index) = batch(cluster1, 'analyse_folders', 1, {job_path});
        disp(sprintf("Submitted job %d: %s", job_index, job_path));
    catch
         warning("Error in job %d", job_index); 
         disp(job(job_index));    
    end
end

%  Wait for all jobs to complete
for job_index = 1:length(job_list)
    wait(job(job_index), 'finished');
end

disp(job(job_index));