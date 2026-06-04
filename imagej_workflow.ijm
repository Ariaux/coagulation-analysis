// ════════════════════════════════════════════════════════════
// Coagulation Assay — ImageJ Automated Workflow
// ════════════════════════════════════════════════════════════
//
// Replicates: Image → Type → 8-bit → Edit → Invert
//             Rectangle ROI → Analyze → Measure → Mean
//
// Usage (ImageJ/Fiji):
//   1. Drag & drop image, or File > Open
//   2. Plugins > Macros > Run... > select this file
//   3. Draw rectangle on the glass slide, click OK in the dialog
//   4. Results table + CSV saved next to the image

// --- Settings ---
outputDir = getDirectory("image") + "results_" + File.getNameWithoutExtension(getTitle()) + "/";
File.makeDirectory(outputDir);

// --- Step 1: 8-bit ---
run("8-bit");

// --- Step 2: Invert (Revert) ---
run("Invert");

// --- Step 3: Rectangle ROI ---
// Let user draw rectangle, then auto-refine to tight bounding box
waitForUser("Draw rectangle around glass slide, then click OK\n(Use the Rectangle tool, drag across the slide)");

// Save ROI
roiPath = outputDir + "roi.zip";
roiManager("Add");
roiManager("Save", roiPath);

// --- Step 4: Measure ---
run("Set Measurements...", "mean area min max std integrated redirect=None decimal=3");
run("Measure");

// --- Extract Mean ---
mean = getResult("Mean", 0);
area = getResult("Area", 0);
std  = getResult("StdDev", 0);
min  = getResult("Min", 0);
max  = getResult("Max", 0);

// --- Save results ---
saveAs("Results", outputDir + "results.csv");

// --- Create summary text ---
summary  = "Image: " + getTitle() + "\n";
summary += "8-bit + Inverted\n";
summary += "Mean: " + mean + "\n";
summary += "Std:  " + std + "\n";
summary += "Min:  " + min + "\n";
summary += "Max:  " + max + "\n";
summary += "Area (px): " + area + "\n";

File.saveString(summary, outputDir + "summary.txt");

// --- Print ---
print("=== Coagulation Analysis Done ===");
print(summary);
print("Results saved to: " + outputDir);
print("");

// --- Cleanup ---
roiManager("Deselect");
roiManager("Delete");

selectWindow("Results");
run("Close");
