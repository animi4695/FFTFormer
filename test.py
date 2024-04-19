import os
import torch
import argparse
from basicsr.models.archs.fftformer_arch import  fftformer
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image as Image
from tqdm import tqdm
import shutil


class DeblurDataset(Dataset):
    def __init__(self, image_dir, transform=None, is_test=False):
        self.image_dir = image_dir
        self.image_list = os.listdir(os.path.join(image_dir, 'input/'))
        self._check_image(self.image_list)
        self.image_list.sort()
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, 'input', self.image_list[idx]))
        label = Image.open(os.path.join(self.image_dir, 'target', self.image_list[idx]))

        if self.transform:
            image, label = self.transform(image, label)
        else:
            image = F.to_tensor(image)
            label = F.to_tensor(label)
        if self.is_test:
            name = self.image_list[idx]
            return image, label, name
        return image, label

    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg']:
                raise ValueError


def test_dataloader(path, batch_size=1, num_workers=0):
    image_dir = os.path.join(path, 'test')
    dataloader = DataLoader(
        DeblurDataset(image_dir, is_test=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader



def main(args):
    # CUDNN
    # cudnn.benchmark = True
    #
    if not os.path.exists('results/' + args.model_name + '/'):
        os.makedirs('results/' + args.model_name + '/')
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    model = fftformer()
    # print(model)
    if torch.cuda.is_available():
        model.cuda()

    _eval(model, args)

    folder = os.path.join(args.data_dir,'test/input')
    empty_inputs(folder)
    folder = os.path.join(args.data_dir,'test/target')
    empty_inputs(folder)


def copy_files(source_dir, dest_dir):
    # Ensure the source directory exists
    if not os.path.exists(source_dir):
        print(f"Source directory '{source_dir}' does not exist.")
        return

    # Ensure the destination directory exists
    if not os.path.exists(dest_dir):
        print(f"Destination directory '{dest_dir}' does not exist. Creating it...")
        os.makedirs(dest_dir)

    # Get a list of files in the source directory
    files = os.listdir(source_dir)

    # Copy each file from the source directory to the destination directory
    for file_name in files:
        source_path = os.path.join(source_dir, file_name)
        dest_path = os.path.join(dest_dir, file_name)
        try:
            shutil.copy2(source_path, dest_path)  # Use copy2 to preserve metadata
        except Exception as e:
            print(f"Error copying '{file_name}': {e}")


    
def empty_inputs(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def _eval(model, args):
    state_dict = torch.load(args.test_model)
    model.load_state_dict(state_dict,strict = True)
    device = torch.device( 'cuda')
    dataloader = test_dataloader(args.data_dir, batch_size=1, num_workers=3)
    torch.cuda.empty_cache()

    model.eval()

    with torch.no_grad():

        # Main Evaluation
        for iter_idx, data in tqdm(enumerate(dataloader)):
            input_img, label_img, name = data

            input_img = input_img.to(device)

            b, c, h, w = input_img.shape
            h_n = (32 - h % 32) % 32
            w_n = (32 - w % 32) % 32
            input_img = torch.nn.functional.pad(input_img, (0, w_n, 0, h_n), mode='reflect')

            pred = model(input_img)
            torch.cuda.synchronize()
            pred = pred[:, :, :h, :w]

            pred_clip = torch.clamp(pred, 0, 1)


            if args.save_image:
                save_name = os.path.join(args.result_dir, name[0])
                pred_clip += 0.5 / 255
                pred = F.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
                pred.save(save_name)

if __name__ == '__main__':

    gblur_types = ['blur_gb_var_1', 'blur_gb_var_3', 'blur_gb_var_5', 'blur_gb_var_7']
    gopro_folders = [ 'GOPR0396_11_00', 'GOPR0410_11_00', 'GOPR0854_11_00', 'GOPR0862_11_00', 'GOPR0868_11_00', 'GOPR0869_11_00', 'GOPR0871_11_00', 'GOPR0881_11_01']
    for gopro_fol in gopro_folders:

        for data in gblur_types:
            parser = argparse.ArgumentParser()
            # copy images to input
            dst = os.path.join('./dataset/gopro/test/input')
            src = os.path.join('../test_with_gaussian_blur/test_with_gaussian_blur/' + gopro_fol + '/' + data)
            copy_files(src, dst)

            # copy images to target
            dst = os.path.join('./dataset/gopro/test/target')
            src = os.path.join('../test/test/' + gopro_fol + '/sharp')
            copy_files(src, dst)

            # Directories
            parser.add_argument('--model_name', default='fftformer', type=str)
            parser.add_argument('--data_dir', type=str, default='./dataset/gopro/')

            # Test
            parser.add_argument('--test_model', type=str, default='./pretrain_model/fftformer_GoPro.pth')
            parser.add_argument('--save_image', type=bool, default=True, choices=[True, False])

            args = parser.parse_args()
            args.result_dir = os.path.join('results/', args.model_name, 'GoPro/' + gopro_fol + '/' + data)
            print(args)
            main(args)
