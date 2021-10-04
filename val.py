import os
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
import tifffile as tiff
import torch
from options.test_options import TestOptions
from models import create_model
from torch.utils.data import DataLoader


class testDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   
        self.A_size = len(self.A_paths)  

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        A = tiff.imread(A_path)
        A = torch.from_numpy(A).unsqueeze(0)
        A = A.type(torch.FloatTensor)

        return {'A': A, 'A_paths': A_path}

    def __len__(self):
        return self.A_size


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    save_dir = opt.results_dir + opt.save_name + '/test_latest/images'
    if os.path.exists(save_dir)==False:
        os.makedirs(save_dir)
    dataset = testDataset(opt)  # create a dataset given opt.dataset_mode and other options
    test_dataloader = DataLoader(dataset,
                        shuffle=True,
                        num_workers=0,
                        batch_size=1)
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    for i, data in enumerate(test_dataloader):
        print('the image '+data['A_paths'][0]+' is processing')
        img = data['A']
        with torch.no_grad():
            img_processed = model.netG_B(img)
        img_processed = img_processed.cpu().squeeze(0).squeeze(0).detach().numpy()
        _, fename = os.path.split(data['A_paths'][0])
        target_dir = save_dir + '/' + fename
        tiff.imwrite(target_dir,img_processed)

