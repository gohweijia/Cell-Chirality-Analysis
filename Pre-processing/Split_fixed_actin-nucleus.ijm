/*
This macro splits file stacks containing only actin and nucleus images into output folders 'actin' and 'nucleus'. 

Macro will iterate over folders in root_dir. Change 'single_directory' to true to analyse only the root 
*/
//  Variables to be changed
var actin_index = 1;  // slice position of actin image
var single_directory = false;  
root_dir = "Z:/perkin_elmer_exports/01Dec2020-fixed cell-exported/";  
//

setBatchMode(true);
if (actin_index == 1)
	var nucleus_index = 2;
else
	var nucleus_index = 1;

if (single_directory == 1)
	split_images(root_dir);
else {
	folders = getFileList(root_dir);
	for (folder=0; folder<folders.length; folder++) {
		subfolder = root_dir + folders[folder];
		print("Processing: ", subfolder);
		split_images(subfolder);
	}
}
print('Done');

function split_images(directory) {
	actin_dir = directory + "actin/";
	nucleus_dir = directory + "nucleus/";  
	File.makeDirectory(actin_dir);
	File.makeDirectory(nucleus_dir);
	
	filelist = getFileList(directory);
	for (i=0; i<filelist.length; i++) {
		file = filelist[i];
		if (endsWith(file, 'tif')) {
			open(directory + file);
			run("Deinterleave", "how=2");
			actin_window = file + " #" + actin_index;
			nucleus_window = file + " #" + nucleus_index;

			// save actin image
			selectWindow(actin_window);
			run("Enhance Contrast", "saturated=0.35");
			run("Apply LUT");
			run("8-bit");
			save(actin_dir + file);

			// save nucleus image
			selectWindow(nucleus_window);
			run("8-bit");
			save(nucleus_dir + file);
		}		
	}
}
