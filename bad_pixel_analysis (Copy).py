import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from matplotlib.backends.backend_pdf import PdfPages
import glob

# Input data file
openfile_path = "/home/czti/bc_evevent_file.txt"
data_file = pd.read_csv(openfile_path, sep=" ", header=None)
n_files = 1  # Adjust based on your needs (length of data_file)

# Sigma clipping threshold
sigma_threshold = 10 # Decide the n-sigma threshold for bad pixel detection

# Output PDF
output_pdf_path = "/home/czti/user_area/anuraag/CZTI_report/CZTI_bad_pixel_aggregate_histograms_3.pdf"
with PdfPages(output_pdf_path) as pdf:
    for i in range(n_files+1):  # Iterate over files
        hdulist = fits.open(data_file[0][i])
        try:
            for j in range(4):  # Iterate over quadrants
                quadrant_name = f"Q{j}"
                data = hdulist['quadrant_name'].data
                
                # Create a figure for each quadrant
                fig, axes = plt.subplots(4, 4, figsize=(15, 15))
                fig.suptitle(f"Quadrant {quadrant_name} - Bad Pixel Analysis", fontsize=16)

                for k in range(16):  # Iterate over detectors
                    det_data = data[data['detID'] == k]
                    bad_pix = []
                    
                    pixel_data = det_data["pixID"]
                    event_counts, _ = np.histogram(pixel_data, bins=20)
                    mean_event_counts, median_event_count, std_event_count = sigma_clipped_stats(event_counts, sigma=4.0)
                    if event_counts > median_event_count + sigma_threshold * std_event_count:
                        bad_pix.append()

                    # Count occurrences of bad pixels
                    unique_numbers = sorted(set(bad_pix))
                    counts = [bad_pix.count(x) for x in unique_numbers]

                    # Plot for each detector
                    ax = axes[k // 4, k % 4]  # Determine subplot position
                    if unique_numbers:
                        ax.bar(unique_numbers, counts, color='r', alpha=0.7)
                    ax.set_title(f"DetID {k}", fontsize=10)
                    ax.set_xlabel("Pixel ID", fontsize=8)
                    ax.set_ylabel("Counts", fontsize=8)
                    ax.tick_params(axis='both', which='major', labelsize=6)

                # Adjust layout and save to PDF
                plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the title
                pdf.savefig(fig)
                plt.close(fig)
        except Exception as e:
            print(f"Error processing file {data_file[0][i]} quadrant {quadrant_name}: {e}")
