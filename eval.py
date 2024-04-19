from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage import io
import os
import lpips
import torch

# Path to the directory containing the model-generated images
generated_images_base_dir = './results/fftformer/GoPro/'
# Path to the directory containing the actual images
actual_images_base_dir = '../test/test/'

# Output file to write SSIM, LPIPS, and PSNR scores
output_file = 'evaluation_scores.txt'

# Initialize dictionary to store total scores and number of pairs for each folder
folder_ssim_scores = {}
folder_lpips_scores = {}
folder_psnr_scores = {}

# Initialize the LPIPS model
lpips_model = lpips.LPIPS(net='alex', version='0.1')

# Open the output file for writing
with open(output_file, 'w') as f:
    # Iterate over each folder in the generated images directory
    for folder_name in os.listdir(generated_images_base_dir):
        generated_images_dir = os.path.join(generated_images_base_dir, folder_name)
        actual_images_dir = os.path.join(actual_images_base_dir, folder_name)
        print("Folder: ", folder_name)
        # List the files in the generated images directory
        # List the files in the actual images directory
        for img_dir in os.listdir(generated_images_dir):
            if img_dir.startswith('blur'):
                print("Sub Folder: ", img_dir)
                generated_dir = os.path.join(generated_images_dir, img_dir)
                
                parts = img_dir.split('_')
                b = ''
                if(len(parts) > 1):
                    b = 'sharp' + '_size_' + parts[2]
                else:
                    b = 'sharp'
                    
                actual_dir = os.path.join(actual_images_dir, b)
                print(generated_dir)
                print(actual_dir)
                generated_image_files = os.listdir(generated_dir)
                actual_image_files = os.listdir(actual_dir)

                # Initialize variables for evaluation for this folder
                total_ssim_score = 0
                total_lpips_score = 0
                total_psnr_score = 0
                num_pairs = 0
                # Calculate scores for each pair of images
                for gen_img_file, actual_img_file in zip(generated_image_files, actual_image_files):
                    # Load the images
                    gen_img = io.imread(os.path.join(generated_dir, gen_img_file))
                    actual_img = io.imread(os.path.join(actual_dir, actual_img_file))
                    gen_img_tensor = torch.tensor(gen_img.transpose(2, 0, 1)).float() / 255.0
                    actual_img_tensor = torch.tensor(actual_img.transpose(2, 0, 1)).float() / 255.0

                    # Calculate the SSIM score
                    ssim_score = compare_ssim(gen_img, actual_img, win_size=3, multichannel=True)

                    # Calculate the LPIPS score
                    lpips_score = lpips_model(gen_img_tensor, actual_img_tensor).item()

                    # Calculate the PSNR score
                    psnr_score = compare_psnr(gen_img, actual_img)

                    # Update total scores and number of pairs
                    total_ssim_score += ssim_score
                    total_lpips_score += lpips_score
                    total_psnr_score += psnr_score
                    num_pairs += 1

                # Calculate the average scores for this folder
                avg_ssim_score = total_ssim_score / num_pairs
                avg_lpips_score = total_lpips_score / num_pairs
                avg_psnr_score = total_psnr_score / num_pairs
                folder_ssim_scores[folder_name] = avg_ssim_score
                folder_lpips_scores[folder_name] = avg_lpips_score
                folder_psnr_scores[folder_name] = avg_psnr_score
                print("avg_ssim_score: ", avg_ssim_score)
                print("avg_lpips_score: ",avg_lpips_score)
                print("avg_psnr_score: ",avg_psnr_score)

                # Write the folder name and corresponding average scores to the output file
                f.write(f'{folder_name} {img_dir}: SSIM: {avg_ssim_score}, LPIPS: {avg_lpips_score}, PSNR: {avg_psnr_score}\n')

# Print the average scores for each folder
for folder, avg_ssim in folder_ssim_scores.items():
    print(f'Average SSIM score for {folder}: {avg_ssim}')

for folder, avg_lpips in folder_lpips_scores.items():
    print(f'Average LPIPS score for {folder}: {avg_lpips}')

for folder, avg_psnr in folder_psnr_scores.items():
    print(f'Average PSNR score for {folder}: {avg_psnr}')