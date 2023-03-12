import numpy as np
import SimpleITK as sitk
import os
from os import listdir
from os.path import isfile, join
import glob
import matplotlib.pyplot as plt
#%matplotlib inline

def createMIP(np_img, slices_num):
    ''' create the mip image from original image, slice_num is the number of 
    slices for maximum intensity projection'''
    np_img = np_img.transpose(1,0,2)
    img_shape = np_img.shape
    np_mip = np.zeros(img_shape)
    for i in range(img_shape[0]):
        start = max(0, i-slices_num)
        np_mip[i,:,:] = np.amin(np_img[start:i+1],0)
    return np_mip.transpose(1,0,2)

def main():
    slices_num = 16
    
    inputfiles = ["../pd_wip/wip_registration_nifti/train","../pd_wip/wip_registration_nifti/test","../pd_wip/pd_nifti_final/train","../pd_wip/pd_nifti_final/test"]
    #outputfiles = ["../pd_wip/wip_registration_nifti/train_mip16/","../pd_wip/wip_registration_nifti/test_mip16/","../pd_wip/pd_nifti_final/train16/",]
    for input in inputfiles:
        #input = "../ARIC/pd_wip/pd_nifti_final/pd_test/"
        #output = "../ARIC/pd_wip/pd_nifti_final/pd_test_mip/"
 
        output = input+str(slices_num)
        
        dirs = listdir(input)
        for dir in dirs:
            dir_path = join(input,dir)
            #files = glob.glob(join(dir_path, '*.nii.gz'))
            #files += glob.glob(join(dir_path,'*.nii'))

            if dir.endswith('nii.gz'):
                print (dir)
                output_dir = join(output,dir)
                if not os.path.exists(output):
                    os.makedirs(output)
                sitk_img = sitk.ReadImage(dir_path)
                np_img = sitk.GetArrayFromImage(sitk_img)
                np_mip = createMIP(np_img,slices_num)
                sitk_mip = sitk.GetImageFromArray(np_mip)
                sitk_mip.SetOrigin(sitk_img.GetOrigin())
                sitk_mip.SetSpacing(sitk_img.GetSpacing())
                sitk_mip.SetDirection(sitk_img.GetDirection())
                writer = sitk.ImageFileWriter()
                #writer.SetFileName(join("out/", 'mip.nii.gz'))
                writer.SetFileName(output_dir)
                writer.Execute(sitk_mip)

if __name__ == '__main__':
    main()  