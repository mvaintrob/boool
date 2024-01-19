import glob
import re
import imageio

# Function to extract the number from the filename
def extract_number(filename):
    return int(re.search(r'hist_(\d+).png', filename).group(1))

# Find all files following the pattern 'hist_<number>.png'
file_list = glob.glob(f'hist_*.png')

# Sort files based on the number in the filename
file_list.sort(key=extract_number)

# Read images
images = [imageio.v2.imread(filename) for filename in file_list]

# Create the GIF
output_gif_path = 'histograms.gif'  # Update this with your desired output path
imageio.mimsave(output_gif_path, images, duration=0.5)
