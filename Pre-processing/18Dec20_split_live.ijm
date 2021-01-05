filelist = newArray(
	"/Volumes/WJ/perkin_elmer_exports/all_live_data/05Jul19_GFP-Actn4_45min/",
	"/Volumes/WJ/perkin_elmer_exports/all_live_data/05Jul19_GFP-Actn4_45min2/"	
);


var num_channels = 1;
var start_index = 0;

// if multiple channels exist
var save_nucleus = true;
var save_third_channel = true;

var actin_num = 2;  // index of actin in multi-channel tif
var nucleus_num = 3;
var third_channel_num = 1;
var third_channel_save_name = "GFP";

setBatchMode(true);



for (file_index = 0; file_index < filelist.length; file_index++) {
   	dir = filelist[file_index];
   	print("Running: ", dir);
   	split_files(dir);
}

function split_files(dir) {
	file_list = getFileList(dir);
	for (j = start_index; j<file_list.length; j++) {
		if (startsWith(file_list[j], "0")==1) {
			if (endsWith(file_list[j], 'tif')==1) {
				base_name = file_list[j];
				file_name = dir + base_name;	
				
				folder_name = substring(base_name, 0, 4);		
				save_dir = dir + folder_name + "/";	
				File.makeDirectory(save_dir);
				print("processing: " + base_name);
				open(file_name);	
				if (num_channels > 1) {
					run("Deinterleave", "how=" + num_channels);	
					actin_filename = base_name +" #" + actin_num;
					
					if (save_nucleus==true) {
						nucleus_filename = base_name + " #" + nucleus_num;
						selectWindow(nucleus_filename);
						run("Enhance Contrast", "saturated=0.35");
						run("8-bit");
						File.makeDirectory(save_dir + "nucleus");
						run("Image Sequence... ", "format=TIFF start=1 save="+save_dir+"nucleus/");
						close();
					}
					if (save_third_channel==true) {
						gfp_filename = base_name + " #" + third_channel_num;
						selectWindow(gfp_filename);
						run("Enhance Contrast", "saturated=0.35");
						run("8-bit");
						File.makeDirectory(save_dir + third_channel_save_name);
						run("Image Sequence... ", "format=TIFF start=1 save="+save_dir+third_channel_save_name);
						close();
					}
				} else {
					actin_filename = base_name;
				}
				
				//actin
				selectWindow(actin_filename);
				run("Enhance Contrast", "saturated=0.35");
				run("8-bit");
				run("Grays");
				File.makeDirectory(save_dir + "actin");
				run("Image Sequence... ", "format=TIFF start=1 save="+save_dir+"actin/");
				//3rd channel
				run("Close All");
				//File.delete(file_name);
			}	
		}	
	}
}
print("Done");