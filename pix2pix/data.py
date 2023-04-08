from os.path import join
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import nibabel as nib
import torchvision.transforms as transforms


def get_dataset(debug=False,norm_type='max',mip_type=False):
    #train_dir = join(root_dir, "train")

    #return DatasetFromFolder(train_dir, direction)
    script_path =os.getcwd()
    print("mip_type",mip_type)
    if  mip_type:
        trainname = 'train16'
    else:
        trainname = 'train'
    target  = "../../ARIC/pd_wip/wip_registration_nifti/"
    source = "../../ARIC/pd_wip/pd_nifti_final/"
    dirtarget = os.path.join(script_path,target+trainname)
    dirsource = os.path.join(script_path, source+trainname)
    n_slices_exclude = 10
    patches_per_set =110
    #path to store the data
    #suffix_npy ="_unet2d_320x320x120(60)(60)_[320x320]_psm9"
    suffix_npy ="_norm_"+norm_type+"_mip_"+str(mip_type)
    if not os.path.exists(os.path.join(script_path, 'xtrain_master_noaug' + suffix_npy + '.npy')):
        print("training data not found, so must create it")
        try:
            #print(dirsource,dirtarget)
            srcfiles, tgtfiles = get_source_and_target_files(dirsource, dirtarget)
            srcfiles.sort()
            tgtfiles.sort()
            #select 10 for tiny training
            if debug:
                srcfiles=srcfiles[0:5]
                tgtfiles=tgtfiles[0:5]
            print("srcfiles size",len(srcfiles))
            print("tgtfiles size",len(tgtfiles))
        except:
            print(dirsource,dirtarget)
            print('could not find', 'source or target files so skipping')
        totalpatches = len(srcfiles) * patches_per_set
        xtrain_master_noaug = np.zeros([totalpatches, 320, 320, 1], dtype=np.float32)
        ytrain_master_noaug = np.zeros([totalpatches, 320, 320, 1], dtype=np.float32)
        slice_count = 0  # counter holding number of slices incorporated into training
        ###############################################################################
        # load .npy files from disk, display central coronal slice, and fill xtrain and ytrain matrices
        ###############################################################################
        slices_per_file = []  # recording the number of slices used for training data extractions

        for m, icode in enumerate(srcfiles):  # loop over volumes to train from
            #print(m,icode)

            #m = 0
            #icode = "F100601-20180814_142803_PD_SPACE_0.5_20180814132547_14.nii.gz"

            print('##############################################')
            print('pd file =>', icode)
            ######################################
            # load numpy arrays, reproject dataset (if needed) and trim and normalize dataset,
            ###################################
            # (320, 128, 320)
            volume1 = load_gz_to_numpy_vol(dirsource, icode, norm_type=norm_type)
            print('wid is =>', tgtfiles[m])
            volume3 = load_gz_to_numpy_vol(dirtarget, tgtfiles[m], norm_type=norm_type)

            print('creating training data set...')
            xtrain_master_noaug[m * patches_per_set:(m + 1) * patches_per_set, :, :,
                   0] = volume1.transpose(2,0,1)[n_slices_exclude:n_slices_exclude+patches_per_set,:,:]
            ytrain_master_noaug[m * patches_per_set:(m + 1) * patches_per_set, :, :,
                    0] = volume3.transpose(2,0,1)[n_slices_exclude:n_slices_exclude+patches_per_set,:,:]

            if m == (len(srcfiles) - 1):  # if last volume, save the training data to disk
                np.save(os.path.join(script_path, 'xtrain_master_noaug' + suffix_npy), xtrain_master_noaug)
                np.save(os.path.join(script_path, 'ytrain_master_noaug' + suffix_npy), ytrain_master_noaug)
    else:
        print("training data found, so just load it")
        # load numpy arrays
        xtrain_master_noaug = np.load(
            os.path.join(script_path, 'xtrain_master_noaug' + suffix_npy + '.npy'))
        ytrain_master_noaug = np.load(
            os.path.join(script_path, 'ytrain_master_noaug' + suffix_npy + '.npy'))
        print('loaded xtrain_master_noaug' + suffix_npy + '.npy' + ' shape: ', xtrain_master_noaug.shape)
        print('loaded ytrain_master_noaug' + suffix_npy + '.npy' + ' shape: ', ytrain_master_noaug.shape)
    
    ytrain_master = np.copy(ytrain_master_noaug)
    xtrain_master = np.copy(xtrain_master_noaug)
    split = int(xtrain_master.shape[0]*0.8)
    inds_all = np.arange(0, split)

    #print('using all data to train network')
    print('use 80% for training, 20% for testing')
    inds_to_train_from = np.copy(inds_all)
    xtrain = xtrain_master[inds_to_train_from, :, :, :]
    ytrain = ytrain_master[inds_to_train_from, :, :, :]
    xtest = xtrain_master[split:,:,:,:]
    ytest = ytrain_master[split:,:,:,:]
    
    return xtrain,ytrain,xtest,ytest


def get_training_set(x,y):

    return MRIDataset(x,y)


def get_test_set(x,y):

    return MRIDataset(x,y)


# helper function that returns source and target tiff file names
def get_source_and_target_files(dirsource, dirtarget):
    """
    # helper function that returns tiff files to process
    inputs:
    dirsource  folder containing input tiff files
    dirtarget  folder containing output tiff files

    outputs:   srcfiles, tgtfiles
    """
    srcfiles = []
    for fname in os.listdir(dirsource):
        if fname.endswith('.gz') > 0 or fname.endswith('.tiff'):
            srcfiles.append(fname)
    tgtfiles = []
    for fname in os.listdir(dirtarget):
        if fname.endswith('.gz') > 0 or fname.endswith('.tiff'):
            tgtfiles.append(fname)
    return srcfiles, tgtfiles

def load_gz_to_numpy_vol(in_dir, in_fname,norm_type='max'):
    path = os.path.join(in_dir, in_fname)
    image_obj = nib.load(path)
    # Extract data as numpy ndarray
    image_data = image_obj.get_fdata()
    
    #load the entire gz
    volume = np.float32(image_data) #(320,128,320)
    volume = volume.transpose(0,2,1) #(320,320,128)
    for i in range(volume.shape[2]):
        # map to [0,1]
        #compare the max value of each slice with the 0.95 percentile value, 0.95 percentile value is more robust.
        #print("max value of slice",i,"is",np.amax(volume[:,:,i]))
        #print("95 percentile value of slice",i,"is",np.percentile(volume[:,:,i],95))
        if norm_type == 'percentile':
            norm_value = np.percentile(volume[:,:,i],95)
        else:
            norm_value = np.amax(volume[:,:,i])
        slice = volume[:,:,i]
        slice = slice / norm_value
        slice = torch.from_numpy(slice).expand(1,320,320)

        #then do normalization
        volume[:,:,i] = transforms.Normalize((0.5,), (0.5,))(slice).squeeze().numpy()
    print("volume's shape",volume.shape)
    return volume


class MRIDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self,x,y):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.y = y.transpose(0,3,1,2) #(600,320,320,1)--> (N, C_in, H, W) (600,1,320,320)
        self.x = x.transpose(0,3,1,2) #(600,320,320,1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()        

        return self.x[idx],self.y[idx]

def FileSave(data, file_path):
    """Save a NIFTI file using given file path from an array
    Using: NiBabel"""
    if(np.iscomplex(data).any()):
        data = abs(data)
    nii = nib.Nifti1Image(data, np.eye(4)) 
    nib.save(nii, file_path)

def main():
    xtrain, ytrain, xtest, ytest = get_dataset(debug=True,norm_type='percentile',mip_type=False)
if __name__ == '__main__':
    main()