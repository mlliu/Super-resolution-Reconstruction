from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import SGD, Adam


from utils import *
import matplotlib.pyplot as plt
# %matplotlib inline

from unet import unet2d

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


"""
key parameters (begin)
##############################################################################
"""

combomatrix = [64, 64, 32, 32, 9, 10000, 10000, 8, False]
'''
 in form [blksz_2d[0],           patch size in row direction (pixel units)
             blksz_2d[1],           patch size in column direction (pixel units)
             stride_2d[0],          stride of patch selection in row direction (pixel units)
             stride_2d[1],          stride of patch selection in column direction (pixel units)
             patch_select_mode,     patch selection mode (see compute_metric_volume_2d() function in utils.py)
             patches_per_set_h,     number of high signal/edge training patches per volume
             patches_per_set_l,     number of low signal training patches per volume
             unet_start_ch,         number of starting channels for the U-Net
             unet_resnet_mode]      residual learning mode for the U-Net to predict the difference; (default == False)
'''

# test mode options
testmode = True  # test by training/predicting the first case only for one reduction factor
testmode_epochs = True  # limits the number of epochs

# basic inputs/parameters
#reduction_list = [4] if testmode else [ 6]  # resolution reduction factors to train/predict
proj_direction = 0  # projection direction for training; 0: no projection, 1: lateral projection, 2: frontal projection
loss_function ='ssim_loss' #'mean_squared_error'  # 'ssim_loss'
sleep_when_done = False  # sleep computer when finished
patches_from_volume = True  # False: patches selected from each slice; True: patches selected from whole volume
data_augm_factor = 1  # data augmentation factor; arbitrary values allowed for 2D
optimizers = 'adam'  # ['adam', 'sgd']
leave_one_out_train = False  # performs training using a leave one out scheme
###############################################################################
# key parameters (end)
# ##############################################################################

patch_select_mode = combomatrix[4]  # set patch selection mode
try:
    patches_per_set_h = combomatrix[5]  # set number of high intensity patches selected from each dataset
    patches_per_set_l = combomatrix[6]  # set number of random intensity patches selected from each dataset
except:
    patches_per_set_h = 5000  # set number of high intensity patches selected from each dataset
    patches_per_set_l = 5000  # set number of random intensity patches selected from each dataset
try:
    unet_resnet_mode = combomatrix[8]
except:
    unet_resnet_mode = False

""
# construct folder name where models are
if loss_function == "mean_squared_error":
    foldersuffix = '_' + str(data_augm_factor) + 'aug_proj' + str(proj_direction) + 'psm' + str(
        patch_select_mode) + "_" + "mse"
else:
    foldersuffix = '_' + str(data_augm_factor)  + 'psm' + str(
        patch_select_mode) + "_" + loss_function
#foldersuffix

""
nepochs = 2 if testmode_epochs else 200  # number of epochs to train for
batch_size_train = 400  # training batch size

blksz_2d = combomatrix[0], combomatrix[1]   # block/patch size in pixels
stride_2d = combomatrix[2], combomatrix[3]  # stride for obtaining training blocks

patches_per_slc_h = 60  # high signal intensity/arterial blocks/patches per slice used for training
patches_per_slc_l = 80  # low signal intensity/background blocks/patches per slice used for training
patches_per_slc = patches_per_slc_h + patches_per_slc_l



""
# U-net related inputs/parameters
try:
    unet_start_ch = b_index[7]
except:
    unet_start_ch = 16  # number of starting channels
unet_depth = 4 if combomatrix[0] ** 0.25 >= 2 else 3  # depth (i.e. # of max pooling steps)
unet_inc_rate = 2  # channel increase rate
unet_dropout = 0.5  # dropout rate
unet_batchnorm = False  # batch normalization mode
unet_residual = False  # residual connections mode
unet_batchnorm_str = 'F' if not unet_batchnorm else 'T'
unet_residual_str = 'F' if not unet_residual else 'T'

n_edge_after_proj = 0  # the edge images not used after proj_direction
patches_per_set = patches_per_set_h + patches_per_set_l

blks_rand_shift_mode = False
training_patience = 3
n_slices_exclude = 0  # number of edge slices to not train from (default value is zero)

subset_train_mode = False  # flag that allows to only to train from a subset of slices from the full volume
subset_train_minslc = 1  # training subset mode - minimum slice
subset_train_maxslc = 500  # training subset mode - maximum slice

# factor for additional in-plane (i.e. x and y) cropping of volumes during training phase
crop_train_x = 1 #0.50 if patches_from_volume else 1
crop_train_y = 1 #0.50 if patches_from_volume else 0.6

""
###############################################################################
# find distinct data sets
###############################################################################
#script_path = os.path.split(os.path.abspath(__file__))[0]
script_path =os.getcwd()
dirtarget = os.path.join(script_path, "../../ARIC/pd_wip/wip_registration_nifti/")
#dirsource = "data/train/CentreSquare15p25Mask"
dirsource = os.path.join(script_path, "../../ARIC/pd_wip/pd_nifti/")
#dirtarget = "data/train/anat1"
try:
    srcfiles, tgtfiles = get_source_and_target_files(dirsource, dirtarget)
    srcfiles.sort()
    tgtfiles.sort()
    #split 5 for test
    #srcfiles=srcfiles[:-5]
    #tgtfiles=tgtfiles[:-5]
    print("srcfiles size",len(srcfiles))
    print("tgtfiles size",len(tgtfiles))
except:
    print(dirsource,dirtarget)
    print('could not find', reduction, 'source or target files so skipping')
    

""
rstr = "res" if unet_resnet_mode else ""
p = patches_per_set, patches_per_set_h, patches_per_set_l if patches_from_volume else patches_per_slc, patches_per_slc_h, patches_per_slc_l

###############################################################################
# check if we've already trained the model; if no, train it, if yes, load it from disk
###############################################################################

# 2D UNet
modelsuffix = "_unet2d" + rstr + "-" + "[" + str(stride_2d[0]) + 'x' + str(
    stride_2d[1]) + "]-psm" + str(patch_select_mode) + "-" + str(unet_start_ch) + "-" + str(
    unet_depth) + "-" + str(unet_inc_rate) + "-" + str(
    unet_dropout) + "-" + unet_batchnorm_str + "-" + unet_residual_str + '-batch' + str(
    batch_size_train)
modelprefix = "model_" + str(blksz_2d[0]) + "x" + str(blksz_2d[1]) + "x" + str(p[0]) + "(" + str(
    p[1]) + ")(" + str(p[2]) + ")x" + str(data_augm_factor)

# check if we need to train more models and set the training_needed_flag,
# as well as return the list for leave one out training mode
outpath = 'train_' + 'unet2d' + rstr + '_' + optimizers  + '_batch' + str(
    batch_size_train) + foldersuffix

#script_path = os.path.split(os.path.abspath(__file__))[0]
dirmodel = os.path.join(script_path, outpath)
if not os.path.exists(dirmodel):
    os.makedirs(dirmodel)

training_needed_flag = should_we_train_network(
    os.path.join(dirmodel, modelprefix + modelsuffix), srcfiles)

""
if training_needed_flag:
    print("model not found for sets", " ".join(str(indices_of_datasets_to_train)), ", so must train it")
    ###############################################################################
    # create training data if not already created
    ###############################################################################
    suffix_npy = "_unet2d" + rstr + "_" + str(blksz_2d[0]) + 'x' + str(blksz_2d[1]) + 'x' + str(
        p[0]) + "(" + str(p[1]) + ")(" + str(p[2]) + ")" + "_[" + str(stride_2d[0]) + 'x' + str(
        stride_2d[1]) + ']' + "_psm" + str(patch_select_mode)
    
    if not os.path.exists(os.path.join(script_path, 'xtrain_master_noaug' + suffix_npy + '.npy')):
        print("training data not found, so must create it")
        ###############################################################################
        # create training arrays
        ###############################################################################
        if patches_from_volume:  # select patches from a complete set of data, not slice by slice
            totalpatches = len(srcfiles) * patches_per_set
        else: # select patches from each slice
            totalpatches = totalnumberofslices * patches_per_slc

        xtrain_master_noaug = np.zeros([totalpatches, blksz_2d[0], blksz_2d[1], 1], dtype=np.float32)
        ytrain_master_noaug = np.zeros([totalpatches, blksz_2d[0], blksz_2d[1], 1], dtype=np.float32)
        
        # count the number of total slices to learn from during training phase
        # (i.e. loop through all data sets except the one that is being reconstructed)
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
            volume1, volume1max = load_tiff_volume_and_scale_si(dirsource, icode, crop_train_x,
                                                                crop_train_y, blksz_2d, proj_direction,
                                                                subset_train_mode, subset_train_minslc,
                                                                subset_train_maxslc)
            print('wid is =>', tgtfiles[m])
            volume3, volume3max = load_tiff_volume_and_scale_si(dirtarget, tgtfiles[m], crop_train_x,
                                                                crop_train_y, blksz_2d, proj_direction,
                                                                subset_train_mode, subset_train_minslc,
                                                                subset_train_maxslc)
            
            print('creating training data set...')
            #if patches_from_volume:  # select patches from each whole dataset
            ######################################
            # select patches from volume,
            ###################################
            # xtrain, patches1, volume1 are source # ytrain, patches3, volume3 are target
            # create metric volume used to select blocks
            vol_metric, metric_operator = compute_metric_volume_2d(volume1, volume3,
                                                                   patch_select_mode, stride_2d,
                                                                  n_slices_exclude)
            slc_train_end = volume1.shape[2] - n_slices_exclude
            if not unet_resnet_mode:
                xtrain_master_noaug[m * patches_per_set:(m + 1) * patches_per_set, :, :,
                0], ytrain_master_noaug[m * patches_per_set:(m + 1) * patches_per_set, :, :,
                    0] = get_patches_within_volume(vol_metric,
                                                   [volume1[:, :, n_slices_exclude:slc_train_end],
                                                    volume3[:, :, n_slices_exclude:slc_train_end]],
                                                   blksz_2d, stride_2d, patches_per_set_h,
                                                   patches_per_set_l, seed=m, shuffleP=False,
                                                   metric_operator=metric_operator)
            else:
                xtrain_master_noaug[m * patches_per_set:(m + 1) * patches_per_set, :, :,
                0], ytrain_master_noaug[m * patches_per_set:(m + 1) * patches_per_set, :, :,
                    0] = get_patches_within_volume(vol_metric, [
                    volume1[:, :, n_slices_exclude:slc_train_end],
                    volume3[:, :, n_slices_exclude:slc_train_end] - volume1[:, :,
                                                                    n_slices_exclude:slc_train_end]],
                                                   blksz_2d,
                                                   stride_2d,
                                                   patches_per_set_h,
                                                   patches_per_set_l, seed=m,
                                                   shuffleP=False,
                                                   metric_operator=metric_operator)

            slice_count = slice_count + volume3.shape[2] - 2 * n_slices_exclude

            slices_per_file.append(volume3.shape[2] - 2 * n_slices_exclude)
            print('ytrain_master_noaug.mean: ', np.mean(ytrain_master_noaug))

            if m == (len(srcfiles) - 1):  # if last volume, save the training data to disk
                np.save(os.path.join(script_path, 'xtrain_master_noaug' + suffix_npy), xtrain_master_noaug)
                np.save(os.path.join(script_path, 'ytrain_master_noaug' + suffix_npy), ytrain_master_noaug)

            print('total files read for training: ', len(srcfiles))
            print('total slices read for training: ', slice_count)
            print('unaugmented patch array size for training: ', xtrain_master_noaug.shape)
            print('slices (or blks) read from all the files: ', slices_per_file)
         ###############################################################################
    # load training data from disk if not already created
    ###############################################################################
    else:
        print("training data found, so just load it")
        # load numpy arrays
        xtrain_master_noaug = np.load(
            os.path.join(script_path, 'xtrain_master_noaug' + suffix_npy + '.npy'))
        ytrain_master_noaug = np.load(
            os.path.join(script_path, 'ytrain_master_noaug' + suffix_npy + '.npy'))
        print('loaded xtrain_master_noaug' + suffix_npy + '.npy' + ' shape: ', xtrain_master_noaug.shape)
        print('loaded ytrain_master_noaug' + suffix_npy + '.npy' + ' shape: ', ytrain_master_noaug.shape)
    print(" ")

    ###############################################################################
    # augment training data by factor of data_augm_factor
    ###############################################################################
    if data_augm_factor > 1:
        print("augmenting data by factor of", data_augm_factor, "...")
        iSlc = 0
        xtrain_master = augment_patches(xtrain_master_noaug, data_augm_factor, iSlc, 180,
                                            (blksz_2d[0] // 4, blksz_2d[1] // 4), 0.4)
        ytrain_master = augment_patches(ytrain_master_noaug, data_augm_factor, iSlc, 180,
                                            (blksz_2d[0] // 4, blksz_2d[1] // 4), 0.4)
        print("augmenting data by factor of", data_augm_factor, "... done")
    else:
        # don't data augment
        ytrain_master = np.copy(ytrain_master_noaug)
        xtrain_master = np.copy(xtrain_master_noaug)
        ytrain = ytrain_master_noaug
        del ytrain_master_noaug
        xtrain = xtrain_master_noaug
        del xtrain_master_noaug

    ###############################################################################
    # define model and print summary
    ###############################################################################

    n_loo_loops = 1  # only one network is trained for all data sets

    if testmode:  # only do first set if we are prototyping
        if set_to_train > 0:
            continue


    inds_all = np.arange(0, xtrain_master.shape[0])

    print('using all data to train network')
    inds_to_train_from = np.copy(inds_all)
    xtrain = xtrain_master[inds_to_train_from, :, :, :]
    ytrain = ytrain_master[inds_to_train_from, :, :, :]

    ###############################################################################
    # define model and print summary
    ###############################################################################
    try:
        del model
    except:
        pass
    model = unet2d((blksz_2d[0], blksz_2d[1], 1), out_ch=1, start_ch=unet_start_ch,
                   depth=unet_depth, inc_rate=unet_inc_rate, activation="relu", dropout=unet_dropout,
                   batchnorm=unet_batchnorm, maxpool=True, upconv=True, residual=unet_residual)
    print(model.summary())

    ###############################################################################
    # compile the model
    ###############################################################################
    opt = Adam()

    if loss_function != 'ssim_loss':
        model.compile(loss=loss_function, optimizer=opt)
    else:
        model.compile(loss=ssim_loss, optimizer=opt, metrics=[ssim_loss, 'accuracy'])

    ###############################################################################
    # checkpointing the model
    ###############################################################################

    filepath = os.path.join(dirmodel,
                            modelprefix + modelsuffix + "-{epoch:02d}-{loss:.6f}-{val_loss:.6f}.hdf5")
    checkpoint1 = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                  mode='min')
    checkpoint2 = EarlyStopping(monitor='val_loss', patience=training_patience, verbose=1, mode='min')
    callbacks_list = [checkpoint1, checkpoint2]

    ###############################################################################
    # fit the model
    ###############################################################################
    print('xtrain size: ', xtrain.shape)
    print('ytrain size: ', ytrain.shape)
    split = int(xtrain.shape[0]*0.8)
    train_dataset = tf.data.Dataset.from_tensor_slices((xtrain[:split,:,:,:], ytrain[:split,:,:,:]))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size_train)
    validation_data = tf.data.Dataset.from_tensor_slices((xtrain[split:,:,:,:], ytrain[split:,:,:,:]))
    validation_data = validation_data.batch(batch_size_train)
    with tf.device('/gpu:0'):
        history = model.fit(train_dataset,validation_data=validation_data, batch_size=max(data_augm_factor, batch_size_train),
                            epochs=nepochs, callbacks=callbacks_list)
        print(history.history.keys())
        print("loss: ", history.history['loss'])
        print("val_loss ", history.history['val_loss'])

    ##################################################################
    # find the the number of the last training epoch written to disk
    ##################################################################
    found = False
    lastepochtofind = len(history.history['loss'])
    print('looking for a file that starts with',
          modelprefix + modelsuffix + "-" + "{0:0=2d}".format(lastepochtofind), 'and ends with .hdf5')
    while not found and lastepochtofind > 0:
        for root, _, files in os.walk(dirmodel):
            for f in files:
                if f.startswith(modelprefix + modelsuffix + "-" + "{0:0=2d}".format(
                        lastepochtofind)) > 0 and f.endswith('.hdf5'):
                    found = True
                    print("last epoch was " + str(lastepochtofind))
                    break
        if not found:
            lastepochtofind -= 1  # reduce epoch number to try to find .hdf5 file if not already found
    if not found:
        print("failed to find most recent good training epoch... this shouldn't happen")
        sys.exit()

    ###############################################################################
    # serialize model to json and write to disk
    ###############################################################################
    model_json = model.to_json()
    with open(os.path.join(dirmodel,
                           modelprefix + modelsuffix  + ".json"),
              "w") as json_file:
        json_file.write(model_json)
    ###############################################################################
    # save weights and write to disk as .h5 file
    ###############################################################################
    model.save_weights(
        os.path.join(dirmodel, modelprefix + modelsuffix  + ".h5"))
    print("saved model to disk")
else:
    pass  # network already trained so do nothing

if sleep_when_done:
    # from: https://stackoverflow.com/questions/37009777/how-to-make-a-windows-10-computer-go-to-sleep-with-a-python-script
    os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")




""
#i=103
#plt.imshow(volume3[:, :, i], cmap='gray')
#plt.axis('off');

""


""

