import tifffile
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping

from tensorflow.keras.optimizers import SGD, Adam


from models import *
from utils import *
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from DataGenerator import DataGenerator
#import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

"""
key simulation parameters (begin)
##############################################################################
"""
#combomatrix = [8, 8, 8, 4, 4, 4, 9, 10000, 10000, 8, False]
combomatrix = [128, 128, 64, 64, 64, 32, 9, 32, 0, 8, False]
'''
 in form [blksz_3d[0],           block size in row direction (pixel units)
             blksz_3d[1],           block size in column direction (pixel units)
             blksz_3d[2],           block size in slice direction (pixel units)
             stride_3d[0],          stride of block selection in row direction (pixel units)
             stride_3d[1],          stride of block selection in column direction (pixel units)
             stride_3d[2],          stride of block selection in slice direction (pixel units)
             patch_select_mode,     block selection mode (see compute_metric_volume_3d() function in utils.py)
             patches_per_set_h,     number of high signal/edge training blocks per volume
             patches_per_set_l,     number of low signal training blocks per volume
             unet_start_ch,         number of starting channels for the U-Net
             unet_resnet_mode]      residual learning mode for the U-Net to predict the difference; (default == False)
'''

# test mode options
testmode = False  # test by training/predicting the first case only for one reduction factor
testmode_epochs = False  # limits the number of epochs

# basic inputs/parameters
#reduction_list = [3] if testmode else [2, 3, 4, 5, 6]  # resolution reduction factors to train/predict
proj_direction = 0  # projection direction for training; 0: no projection, 1: lateral projection, 2: frontal projection
loss_function = 'ssim_loss'  # 'ssim_loss'
sleep_when_done = False if testmode else True  # sleep computer when finished
patches_from_volume = True  # False: patches selected from each slice; True: patches selected from whole volume
data_augm_factor = 1  # 1, 2, 4, or 8 are options for residual net
#data_augm_factor = set_augmentation_factor_for_3D_net(data_augm_factor)  # set augmentation factor to 1,2,4 or 8
optimizers = 'adam'  # ['adam', 'sgd']
leave_one_out_train = False  # performs training using a leave one out scheme
lr = 0.0001 #learning rate
scale = 0  # scale for dataloader
###############################################################################
# key simulation parameters (end)
# ##############################################################################

#for b_index in combomatrix:  # loop over reconstructions to perform
# input channels for UNets
input_ch = combomatrix[2]

patch_select_mode = combomatrix[6]  # set patch selection mode
try:
    patches_per_set_h = combomatrix[7]  # set number of high intensity patches selected from each dataset
    patches_per_set_l = combomatrix[8]  # set number of random intensity patches selected from each dataset
except:
    patches_per_set_h = 500  # set number of high intensity patches selected from each dataset
    patches_per_set_l = 500  # set number of random intensity patches selected from each dataset
patches_per_set = patches_per_set_h + patches_per_set_l
try:
    unet_resnet_mode = combomatrix[10]
except:
    unet_resnet_mode = False

# construct folder name where models are
if loss_function == "mean_squared_error":
    foldersuffix = '_' + str(data_augm_factor) + 'aug_proj' + str(proj_direction) + 'psm' + str(
        patch_select_mode) + "_" + "mse"
else:
    foldersuffix = '_' + str(data_augm_factor) + 'aug_proj' + str(proj_direction) + 'psm' + str(
        patch_select_mode) + "_" + loss_function

#for optim in optimizers:  # loop through optimizers
nepochs = 2 if testmode_epochs else 200  # number of epochs to train for
batch_size_train = 12 #if b_index[0] == 32 else 32 // b_index[0] * 20

blksz_3d = combomatrix[0], combomatrix[1], combomatrix[2]  # block size in pixels that is used to train the model
stride_3d = combomatrix[3], combomatrix[4], combomatrix[5]  # stride size in pixels that is used to train the model

# U-net related inputs/parameters
try:
    unet_start_ch = combomatrix[9]
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

blks_rand_shift_mode = False
training_patience = 3
n_slices_exclude = 4  # number of edge slices to not train from (default value is zero)

subset_train_mode = False  # flag that allows to only to train from a subset of slices from the full volume
subset_train_minslc = 1  # training subset mode - minimum slice
subset_train_maxslc = 500  # training subset mode - maximum slice

# factors for additional in-plane (i.e. x and y) cropping of volumes during training phase
crop_train_x = 1 #0.50 if patches_from_volume else 1
crop_train_y = 1 #0.50 if patches_from_volume else 0.6

###############################################################################
# find distinct data sets
# ##############################################################################
script_path = os.path.split(os.path.abspath(__file__))[0]
dirtarget = os.path.join(script_path,"../../pd_wip/wip_registration_nifti/train")
dirsource = os.path.join(script_path, "../../pd_wip/pd_nifti_final/train")
try:
    srcfiles, tgtfiles = get_source_and_target_files(dirsource, dirtarget)
    srcfiles.sort()
    tgtfiles.sort()
    #select 10 for tiny training
    srcfiles=srcfiles[0:200]
    tgtfiles=tgtfiles[0:200]
    print("srcfiles size",len(srcfiles))
    print("tgtfiles size",len(tgtfiles))
except:
    print('could not find', reduction, 'source or target files so skipping')

###############################################################################
# check if we've already trained the model; if no, train it, if yes, load it from disk
# ##############################################################################
rstr = "res" if unet_resnet_mode else ""
modelsuffix = "_unet3d" + rstr + "-" + "[" + str(stride_3d[0]) + 'x' + str(stride_3d[1]) + 'x' + str(
    stride_3d[2]) + "]-psm" + str(patch_select_mode) + "-" + str(unet_start_ch) + "-" + str(
    unet_depth) + "-" + str(unet_inc_rate) + "-" + str(
    unet_dropout) + "-" + unet_batchnorm_str + "-" + unet_residual_str + '-batch' + str(batch_size_train)
modelprefix = "model_" + str(blksz_3d[0]) + "x" + str(blksz_3d[1]) + "x" + str(blksz_3d[2]) + "x" + str(
    patches_per_set) + "(" + str(patches_per_set_h) + ")(" + str(patches_per_set_l) + ")x" + str(
    data_augm_factor)

# check if we need to train more models and set the training_needed_flag, as well as return the list for leave one out training mode
outpath = 'train_' + 'unet3d' + rstr + '_' + optimizers  + '_batch' + str(
    batch_size_train) + foldersuffix

script_path = os.path.split(os.path.abspath(__file__))[0]
dirmodel = os.path.join(script_path, outpath)
if not os.path.exists(dirmodel):
    os.makedirs(dirmodel)

training_needed_flag = should_we_train_network(
    os.path.join(dirmodel, modelprefix + modelsuffix), srcfiles)

if training_needed_flag:
    print("model not found for sets", " so must train it")
    ###############################################################################
    # create training data if not already created
    ###############################################################################
    suffix_npy = "_unet3d" + rstr + "_3dto3d_" + str(blksz_3d[0]) + "x" + str(blksz_3d[1]) + "x" + str(
        blksz_3d[2]) + "_" + str(patches_per_set) + "(" + str(patches_per_set_h) + ")(" + str(
        patches_per_set_l) + ")"  + "_psm" + str(
        patch_select_mode)
    if not os.path.exists(os.path.join(script_path, 'xtrain_master_noaug' + suffix_npy + '.npy')):
        print("training data not found, so must create it")
        ###############################################################################
        #create training arrays
        ###############################################################################
        totalpatches = len(srcfiles) * patches_per_set

        xtrain_master_noaug = np.zeros([totalpatches, blksz_3d[0], blksz_3d[1], input_ch, 1],
                                       dtype=np.float32)
        ytrain_master_noaug = np.zeros([totalpatches, blksz_3d[0], blksz_3d[1], input_ch, 1],
                                       dtype=np.float32)

                    # count the number of total slices to learn from during training phase (i.e. loop through all data sets except the one that is being reconstructed)
        slice_count = 0  # counter holding number of slices incorporated into training
        ###############################################################################
        # load .npy files from disk, display central coronal slice, and fill xtrain and ytrain matrices
        ###############################################################################
        slices_per_file = []  # recording the number of slices used for training data extractions
        if proj_direction > 0:
            n_slices_exclude = n_edge_after_proj

        for m, icode in enumerate(srcfiles):  # loop over volumes to train from
            #print('##############################################')
            #print('pd file is =>', icode)
            ###################################
            # load numpy arrays, reproject dataset (if needed) and trim and normalize dataset,
            ###################################
            volume1, volume1max = load_tiff_volume_and_scale_si(dirsource, icode, crop_train_x,
                                                                crop_train_y, blksz_3d[:2], proj_direction,
                                                                subset_train_mode, subset_train_minslc,
                                                                subset_train_maxslc)
            
            volume1 = volume1[:,:,n_slices_exclude:-n_slices_exclude]
            #print('wid is =>', tgtfiles[m])
            volume3, volume3max = load_tiff_volume_and_scale_si(dirtarget, tgtfiles[m], crop_train_x,
                                                                crop_train_y, blksz_3d[:2], proj_direction,
                                                                subset_train_mode, subset_train_minslc,
                                                                subset_train_maxslc)
            volume3 = volume3[:,:,n_slices_exclude:-n_slices_exclude]
            #print("volume1.shape: ",volume1.shape)
            #print('creating training data set...')
            # prepare training data for 3D Unet or 3D Residual net
            # 3D Unet
            # create metric volume used to select blocks
            vol_metric, metric_operator, nregions = compute_metric_volume_3d(volume1, volume3,
                                                                             patch_select_mode, stride_3d)

            # create the training input (xtrain_master...) and output (ytrain_master...) 3D blocks
            if not unet_resnet_mode:
                xtrain_master_noaug[m * patches_per_set:(m + 1) * patches_per_set, :, :, :,
                0], ytrain_master_noaug[m * patches_per_set:(m + 1) * patches_per_set, :, :, :,
                    0], blockvolmap = get_blocks_within_volume(vol_metric, [volume1, volume3], blksz_3d,
                                                               stride_3d, patches_per_set_h,
                                                               n_lower=patches_per_set_l, seed=m,
                                                               shuffleP=blks_rand_shift_mode,
                                                               metric_operator=metric_operator,
                                                               nregions=nregions, return_bvm=True)
            else:
                xtrain_master_noaug[m * patches_per_set:(m + 1) * patches_per_set, :, :, :,
                0], ytrain_master_noaug[m * patches_per_set:(m + 1) * patches_per_set, :, :, :,
                    0], blockvolmap = get_blocks_within_volume(vol_metric,
                                                               [volume1, volume3 - volume1],
                                                               blksz_3d,
                                                               stride_3d,
                                                               n_larger=patches_per_set_h,
                                                               n_lower=patches_per_set_l,
                                                               seed=m,
                                                               shuffleP=blks_rand_shift_mode,
                                                               metric_operator=metric_operator,
                                                               nregions=nregions,
                                                               return_bvm=True)
            #print(np.amax(blockvolmap), np.amin(blockvolmap))
            #blockvolmap = np.uint16(np.round(
            #    np.float(volume3max / np.amax(blockvolmap)) * np.moveaxis(blockvolmap, -1,
            #                                                              0)))  # move slices from 3rd dimension to 1st dimension
            #tifffile.imsave(
            #    'blockvolmap' + suffix_npy + '_' + str(m) + "_" + str(nregions) + "regions.tiff",
            #    blockvolmap, compress=6)
            slices_per_file.append(patches_per_set)
            #print('train data fitted: m, m*patches_per_set:(m+1)*patches_per_set: ', m, m * patches_per_set,
            #      (m + 1) * patches_per_set)

            #if m == (len(srcfiles) - 1):  # if last volume, save the training data to disk
            #    np.save(os.path.join(script_path, 'xtrain_master_noaug' + suffix_npy), xtrain_master_noaug)
            #    np.save(os.path.join(script_path, 'ytrain_master_noaug' + suffix_npy), ytrain_master_noaug)

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
    #  flip the data in y, x, an z directions, augmenting by factors of 2, 4, or 8
    ###############################################################################
    '''
    if data_augm_factor >= 2:
        xtrain_master = np.concatenate(
            (xtrain_master_noaug, np.flip(xtrain_master_noaug, axis=1)))  # flip vertically (y)
        ytrain_master = np.concatenate((ytrain_master_noaug, np.flip(ytrain_master_noaug, axis=1)))
    if data_augm_factor >= 4:
        xtrain_master = np.concatenate(
            (xtrain_master, np.flip(xtrain_master, axis=2)))  # flip horizontally (x)
        ytrain_master = np.concatenate((ytrain_master, np.flip(ytrain_master, axis=2)))
    if data_augm_factor >= 8 and len(xtrain_master_noaug.shape) == 4:
        xtrain_master = np.concatenate((xtrain_master, np.flip(xtrain_master, axis=3)))  # flip over (z)
        ytrain_master = np.concatenate((ytrain_master, np.flip(ytrain_master, axis=3)))
    else:  # no augmentation
        #ytrain_master = np.copy(ytrain_master_noaug)
        #xtrain_master = np.copy(xtrain_master_noaug)
        ytrain_master = ytrain_master_noaug
        xtrain_master = xtrain_master_noaug
    '''
    shape_xtrain_master_noaug = xtrain_master_noaug.shape
    #del ytrain_master_noaug
    #del xtrain_master_noaug

    ###############################################################################
    # define model and print summary
    ###############################################################################

    n_loo_loops = 1  # only one network is trained for all data sets

    #inds_all = np.arange(0, xtrain_master.shape[0] * data_augm_factor)
    split = int(xtrain_master_noaug.shape[0]*0.8)
    #inds_all = np.arange(0, split)
    print('use 80% for training, 20% for testing')
    #inds_to_train_from = np.copy(inds_all)
    xtrain = xtrain_master_noaug[:split, :, :, :]
    ytrain = ytrain_master_noaug[:split, :, :, :]
    xtest = xtrain_master_noaug[split:,:,:,:]
    ytest = ytrain_master_noaug[split:,:,:,:]

    ###############################################################################
    # define model and print summary
    ###############################################################################
    try:
        del model
    except:
        pass
    # 3D Unet
    model = unet3d((blksz_3d[0], blksz_3d[1], blksz_3d[2], 1),
                   out_ch=input_ch,
                   start_ch=unet_start_ch,
                   depth=unet_depth,
                   inc_rate=unet_inc_rate,
                   activation="relu",
                   dropout=unet_dropout,
                   batchnorm=unet_batchnorm,
                   maxpool=True,
                   upconv=False,
                   residual=unet_residual,
                   zdim=input_ch,
                   true_unet=True)
    print(model.summary())

    ###############################################################################
    # compile the model
    ###############################################################################
    
    opt = Adam(learning_rate=lr)
 
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
    ##############################################################################
    #official data generator
    ##############################################################################
    split = int(xtrain.shape[0]*0.8)
    
    #datagen = ImageDataGenerator(rescale=scale)
    #train_iterator = datagen.flow(xtrain[:split,:,:,:,:], ytrain[:split,:,:,:,:], batch_size=batch_size_train)
    #test_iterator = datagen.flow(xtrain[split:,:,:,:,:], ytrain[split:,:,:,:,:], batch_size=batch_size_train)
    ###############################################################################
    train_iterator = DataGenerator(xtrain, ytrain, batch_size=batch_size_train)
    test_iterator = DataGenerator(xtest, ytest, batch_size=batch_size_train)
    # fit the model
    ###############################################################################
    print('xtrain size: ', xtrain.shape)
    print('ytrain size: ', ytrain.shape)
    with tf.device('/gpu:0'):
        #history = model.fit(xtrain, ytrain, batch_size=max(data_augm_factor, batch_size_train),
        #                    epochs=nepochs, callbacks=callbacks_list, validation_split=0.2,
        #                    shuffle=True)
        history = model.fit(train_iterator, steps_per_epoch=len(train_iterator),validation_data = test_iterator, 
                            validation_steps = len(test_iterator) ,epochs=nepochs,callbacks=callbacks_list)
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
                               modelprefix + modelsuffix + ".json"),
                  "w") as json_file:
            json_file.write(model_json)
        ###############################################################################
        # save weights and write to disk as .h5 file
        ###############################################################################
        model.save_weights(
            os.path.join(dirmodel, modelprefix + modelsuffix +  ".h5"))
        print("saved model to disk")
else:
    pass  # network already trained so do nothing



#if sleep_when_done:
    # from: https://stackoverflow.com/questions/37009777/how-to-make-a-windows-10-computer-go-to-sleep-with-a-python-script
#    os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
