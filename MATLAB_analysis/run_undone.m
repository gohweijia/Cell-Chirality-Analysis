%  Put paths in cell array 'paths'. 
ncpus = feature('numcores')

txt_path = 'paths.txt';
paths = textread(txt_path,'%s','delimiter','\n');
paths(ismember(paths,'')) = [];  % Remove empty line
paths(contains(paths,'%')) = [];  % Remove comments

global nb_thresh;
global min_length_um;
global pixel_size;
pixel_size = 0.138502;            % micrometer
min_length_um = 1;
nb_thresh = -0.2;
%cluster1 = parcluster;


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


cluster1 = parcluster;
for job_index = 1:length(job_list)
    job_path = job_list{job_index};
    if job_index <= ncpus 
        job(job_index) = batch(cluster1, 'analyse_folders', 1, {job_path});
        disp(sprintf("Submitted job %d: %s", job_index, job_path));
    elseif string(job(job_index - ncpus).State) ~= 'finished'
        disp('Waiting for job ' + string(job_index-ncpus));
        wait(job(job_index - ncpus));
        job(job_index) = batch(cluster1, 'analyse_folders', 1, {job_path});
        disp(sprintf("Submitted job %d: %s", job_index, job_path));
    end
end