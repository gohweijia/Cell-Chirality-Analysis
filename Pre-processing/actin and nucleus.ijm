var actin_index = 1;
setBatchMode(true);

if (actin_index == 1)
	var nucleus_index = 2;
else
	var nucleus_index = 1;
	
directory = "Z:/perkin_elmer_exports/01Dec2020-fixed cell-exported/";
folders = getFileList(directory);
for (folder=0; folder<folders.length; folder++) {
	subfolder = directory + folders[folder];
	print(subfolder);
	process_images(subfolder);
	
}
print('Done');
function process_images(directory) {
	actin_dir = directory + "actin/";
	gfp_dir = directory + "nucleus/";
    
	File.makeDirectory(actin_dir);
	File.makeDirectory(gfp_dir);
	
	filelist = getFileList(directory);
	for (i=0; i<filelist.length; i++) {
		file = filelist[i];
		if (endsWith(file, 'tif')) {
			open(directory + file);
			run("Deinterleave", "how=2");
			
			// second channel Pax, third channel Actin
			actin_window = file + " #" + actin_index;
			gfp_window = file + " #" + nucleus_index;



//			// For when there is time series 
//			selectWindow(actin_window);
//			run("Duplicate...", " ");
//			actin_window = actin_window + "-1";
//			selectWindow(gfp_window);
//			run("Duplicate...", " ");
//			gfp_window = gfp_window + "-1";

			
	
			// save actin image
			selectWindow(actin_window);
			run("Enhance Contrast", "saturated=0.35");
			run("Apply LUT");
			run("8-bit");
			save(actin_dir + file);
//			

//	
			// save GFP image
			selectWindow(gfp_window);
			//run("8-bit");
			save(gfp_dir + file);
			
			

		}
		
	}


}
