from os.path import join
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import nibabel as nib


def get_dataset():
    #train_dir = join(root_dir, "train")

    #return DatasetFromFolder(train_dir, direction)
    script_path =os.getcwd()
    dirtarget = os.path.join(script_path, "../../pd_wip/wip_registration_nifti/train")
    #dirsource = "data/train/CentreSquare15p25Mask"
    dirsource = os.path.join(script_path, "../../pd_wip/pd_nifti_final/train")
    #dirtarget = "data/train/anat1"
    n_slices_exclude = 4
    patches_per_set =120
    #path to store the data
    suffix_npy ="_unet2d_320x320x120(60)(60)_[320x320]_psm9"
    if not os.path.exists(os.path.join(script_path, 'xtrain_master_noaug' + suffix_npy + '.npy')):
        print("training data not found, so must create it")
        try:
            srcfiles, tgtfiles = get_source_and_target_files(dirsource, dirtarget)
            srcfiles.sort()
            tgtfiles.sort()
            #select 10 for tiny training
            #srcfiles=srcfiles[0:5]
            #tgtfiles=tgtfiles[0:5]
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
            volume1 = load_tiff_volume_and_scale_si(dirsource, icode)
            print('wid is =>', tgtfiles[m])
            volume3 = load_tiff_volume_and_scale_si(dirtarget, tgtfiles[m])

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


def load_tiff_volume_and_scale_si(in_dir, in_fname):
    """
    :param in_dir: input directory
    :param in_fname: input filename
    :param crop_x: crop factor in x direction
    :param crop_y: crop factor in y direction
    :param blksz: patch size (i.e. the length or height of patch, either integer or tuple)
    :param proj_direction: projection direction
    :param subset_train_mode: boolean subset mode flag
    :param subset_train_minslc: if loading a subset, min slice index to load
    :param subset_train_maxslc: if loading a subset, max slice index to load
    :return:    vol (tiff volume scaled between 0 and 1), maxsi (maximum signal intensity of original volume)
    """

    #vol = load_tiff_to_numpy_vol(os.path.join(in_dir, in_fname), subset_train_mode, subset_train_minslc,
    #                             subset_train_maxslc)
    vol = load_gz_to_numpy_vol(os.path.join(in_dir, in_fname))
    print("volume's shape",vol.shape)

    # adjust x and y dimensions of volume to divide evenly into blksz
    # while cropping using 'crop_train' to increase speed and avoid non-pertinent regions

    #vol = crop_volume_in_xy_and_reproject_2D(vol, crop_x, crop_y, blksz, proj_direction)
    #vol = np.float32(vol)
    if len(np.argwhere(np.isinf(vol))) > 0:
        for xyz in np.argwhere(np.isinf(vol)):
            vol[xyz[0], xyz[1], xyz[2]] = 0
    print('max signal value is', np.amax(vol))
    print('min signal value is', np.amin(vol))
    # normalize volumes to have range of 0 to 1

    vol = np.float32(vol / np.amax(vol))  # volume1 is source

    return vol


def load_gz_to_numpy_vol(path):
    image_obj = nib.load(path)
    # Extract data as numpy ndarray
    image_data = image_obj.get_fdata()
    
    #load the entire gz
    volume = np.float32(image_data)
    volume = volume.transpose(0,2,1)
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
