import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
import os

# Input data file
openfile_path = "/home/czti/bc_evevent_file.txt"
#output directory
output_dir = "/home/czti/user_area/anuraag/CZTI_report/"

data_file = pd.read_csv(openfile_path, sep=" ", header=None)
n_files = 1  # Adjust based on your needs (length of data_file)
hdulist = fits.open(data_file[0][0])
selected_quad = 0 # Quadrant to process (0-3)
for selected_quad in range(selected_quad, selected_quad + 4, 1):
    quadrant = f'Q{selected_quad}'
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

    # Output CSV file
    output_csv_path = f"{output_dir}bad_pixels_quad{quadrant}.csv"

    # Prepare the CSV file (write header once)
    # with open(output_csv_path, 'w') as f:
    #     f.write("Quadrant,detID,Bad Pixels,Counts,Reference_counts,noisy_pix_memory_size\n")
    outfile = open(output_csv_path, 'w')
    outfile.write("quad,detID,pixID,count,frac_counts\n")
    # Get data for the selected quadrant
    data = hdulist[quadrant].data

    # Create a figure with 4x4 subplots
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(16, 16))
    fig.suptitle(f"Counts of Pixels for detID 0-15 in {quadrant}", fontsize=16)

    # Iterate over detIDs and process
    for i in range(16):
        ax = axes[i // 4, i % 4]  # Determine the subplot location
        det_id = data[data['detID'] == i]
        pix_data = det_id['pixID']
        
        # Finding the mean and standard deviation of the pixel counts
        pix_counts = np.bincount(pix_data, minlength=256)
        mean_pix_counts, median_pix_counts, std_pix_counts = sigma_clipped_stats(pix_counts, sigma=3)
        
        # Identify bad pixels
        bad_pixels = [idx for idx, val in enumerate(pix_counts) if val > mean_pix_counts + 100 * std_pix_counts]
        #print(f"detID: {i}, Bad Pixels: {bad_pixels}, Counts: {pix_counts[bad_pixels]}, Mean: {mean_pix_counts}, Median: {median_pix_counts}, Std: {std_pix_counts}")
        memory_used_by_noisy_pixels = np.round(pix_counts[bad_pixels] / np.sum(pix_counts) * 100) 
        print(f"detID: {i}, Bad Pixels: {bad_pixels}, Counts: {pix_counts[bad_pixels]}, Fraction of counts from noisy pixels: {memory_used_by_noisy_pixels} %")
        for pix in bad_pixels:
            frac_counts = pix_counts[pix] / np.sum(pix_counts)
            print(f"{selected_quad}, {i}, {pix}, {pix_counts[pix]}, {frac_counts:0.3f}")
        
        outfile.write(f"{selected_quad}, {i}, {pix}, {pix_counts[pix]}, {frac_counts:0.3}\n")
        # Plot histogram for the detector
        ax.hist(pix_data, bins=256)
        ax.set_yscale('log')
        ax.set_xlabel("Pixel Number")
        ax.set_ylabel("Counts (log scale)")
        ax.set_title(f"detID: {i}", fontsize=10)
    outfile.close()

# Adjust layout to prevent overlap and save the figure
plt.tight_layout(rect=[0, 0, 1, 0.96])
output_pdf_path = f"{output_dir}CZTI_pixel_counts_{quadrant}.png"
plt.savefig(output_pdf_path, dpi=300)
plt.close()

print(f"Bad pixel data saved to {output_csv_path}")
print(f"Pixel count plots saved to {output_pdf_path}")
