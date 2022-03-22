import numpy as np 
import lasagne
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers import PadLayer
from lasagne.layers import ExpressionLayer
from lasagne.layers import NonlinearityLayer
from lasagne.nonlinearities import softmax, rectify
from lasagne.layers import batch_norm
from lasagne.layers import Layer
from scipy.spatial.distance import cdist
from sklearn.decomposition import IncrementalPCA
import os
import cv2
import cPickle
###################### Load the data #######################

def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def load_data(samples_pr_cl_val, dim, nb_cl):
    print "loading ./data/auxiliary_cifar100_" + str(nb_cl) + "_dimension" + str(dim)
    xs = []
    ys = []
    d = unpickle('./data/auxiliary_cifar100_' + str(nb_cl) + '_dimension' + str(dim))
    d['data'] = np.array(d['data'])
    d['ori_data'] = np.array(d['ori_data'])
    xs.append(d['ori_data'].reshape(d['ori_data'].shape[0],3,32,32))
    ys.append(d['fine_labels'])
    xs.append(d['data'].reshape(d['data'].shape[0],3,32,32))
    ys.append(d['fine_labels'])
    print d['ori_data'].shape
    print d['data'].shape
    d = unpickle('./data/cifar-100-python/test')
    xs.append(d['data'].reshape(d['data'].shape[0],3,32,32))
    ys.append(d['fine_labels'])
    print d['data'].shape
   
    x = np.concatenate(xs)/np.float32(255)
    y = np.concatenate(ys)
    #x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
    #x = x.reshape((x.shape[0], 32, 32, 3)).transpose(0,3,1,2)
    # subtract per-pixel mean
    pixel_mean = np.mean(x[0:50000],axis=0)
    x -= pixel_mean
    # Create Train/Validation set
    eff_samples_cl = 500-samples_pr_cl_val
    X_train = np.zeros((eff_samples_cl*100,3,32, 32))
    Y_train = np.zeros(eff_samples_cl*100)
    X_produce = np.zeros((eff_samples_cl*100,3,32, 32))
    Y_produce = np.zeros(eff_samples_cl*100)
    for i in range(100):
        index_y=np.where(y[0:50000]==i)[0]
        X_train[i*eff_samples_cl:(i+1)*eff_samples_cl] = x[index_y[0:eff_samples_cl],:,:,:]
        Y_train[i*eff_samples_cl:(i+1)*eff_samples_cl] = y[index_y[0:eff_samples_cl]]
    for i in range(100):
        index_y=np.where(y[50000:100000]==i)[0]
        X_produce[i*eff_samples_cl:(i+1)*eff_samples_cl] = x[index_y[0:eff_samples_cl] + 50000,:,:,:]
        Y_produce[i*eff_samples_cl:(i+1)*eff_samples_cl] = y[index_y[0:eff_samples_cl] + 50000]
    
    X_test  = x[100000:,:,:,:]
    Y_test  = y[100000:]
    assert (Y_train[:50000] == Y_produce[:50000]).all()
    return dict(
        X_train = lasagne.utils.floatX(X_train[:50000]),
        Y_train = Y_train[:50000].astype('int32'),
        X_produce = lasagne.utils.floatX(X_produce),
        Y_produce = Y_produce.astype('int32'),
        X_test  = lasagne.utils.floatX(X_test),
        Y_test  = Y_test.astype('int32'),)


def save(data, file_name):
    cPickle.dump(data,open(file_name, "wb"))
    return
###################### Prepare the low-fidelity auxiliary data #######################
def produce_data_pair(dim, nb_cl):
    ori_d = unpickle('./data/cifar-100-python/train')
    order = np.arange(100)
    filename = "./data/auxiliary_cifar100_" + str(nb_cl) + "_dimension" + str(dim)

    if os.path.exists(filename):
        print("auxiliary data already exists")
        return
    ori_xa = np.array(ori_d['data'])
    ori_ya = np.array(ori_d['fine_labels'])
    
    for i in range(100):
        index = np.where(ori_ya[0:50000]==order[i])[0]
        if i==0:
            xa = ori_xa[index[0:500]]
            ya = ori_ya[index[0:500]]
    
        else:
            xa = np.append(xa, ori_xa[index[0:500]], axis=0)
            ya = np.append(ya, ori_ya[index[0:500]], axis=0)
    incpca = IncrementalPCA(n_components=dim)
    
    for i in range((100/nb_cl)):
        num_per_step = 50000 /(100/nb_cl)
        incpca.partial_fit(xa[i*num_per_step:(i+1)*num_per_step])
        
    newxai = incpca.transform(xa)    
    newxa = incpca.inverse_transform(newxai)
    
    newxa = np.around(newxa)
    ind = np.where(newxa > 255)
    newxa[ind] = 255
    ind = np.where(newxa < 0)
    newxa[ind] = 0
    newxa = newxa.astype(np.uint8)
    
    #store the original and the corresponding auxiliary data
    newd = {}
    newd['data'] = newxa.tolist()
    newd['ori_data'] = xa.tolist()
    newd['fine_labels'] = ya.tolist()
    
    save(newd, filename)
    print("Auxiliary data prepare done")




###################### Build the neural network model #######################

def build_cnn(input_var=None, n=5):
    # This block of code for the architecture and the data augmentation is inspired from Lasagne recipe code : https://github.com/Lasagne/Recipes/blob/master/papers/deep_residual_learning/Deep_Residual_Learning_CIFAR-10.py
    # create a residual learning building block with two stacked 3x3 convlayers as in paper
    def residual_block(l, increase_dim=False, projection=False,last=False):
        input_num_filters = l.output_shape[1]
        if increase_dim:
            first_stride = (2,2)
            out_num_filters = input_num_filters*2
        else:
            first_stride = (1,1)
            out_num_filters = input_num_filters
        
        stack_1 = batch_norm(ConvLayer(l, num_filters=out_num_filters, filter_size=(3,3), stride=first_stride, nonlinearity=rectify, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
        stack_2 = batch_norm(ConvLayer(stack_1, num_filters=out_num_filters, filter_size=(3,3), stride=(1,1), nonlinearity=None, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
        
        # add shortcut connections
        if increase_dim:
            if projection:
                # projection shortcut, as option B in paper
                projection = batch_norm(ConvLayer(l, num_filters=out_num_filters, filter_size=(1,1), stride=(2,2), nonlinearity=None, pad='same', b=None, flip_filters=False))
                if last:
                    block = ElemwiseSumLayer([stack_2, projection])
                else:
                    block = NonlinearityLayer(ElemwiseSumLayer([stack_2, projection]),nonlinearity=rectify)
            else:
                # identity shortcut, as option A in paper
                identity = ExpressionLayer(l, lambda X: X[:, :, ::2, ::2], lambda s: (s[0], s[1], s[2]//2, s[3]//2))
                padding = PadLayer(identity, [out_num_filters//4,0,0], batch_ndim=1)
                if last:
                    block = ElemwiseSumLayer([stack_2, padding])
                else:
                    block = NonlinearityLayer(ElemwiseSumLayer([stack_2, padding]),nonlinearity=rectify)
        else:
            if last:
                block = ElemwiseSumLayer([stack_2, l])
            else:
                block = NonlinearityLayer(ElemwiseSumLayer([stack_2, l]),nonlinearity=rectify)
        
        return block
    
    # Building the network
    l_in = InputLayer(shape=(None, 3, 32, 32), input_var=input_var)
    
    # first layer, output is 16 x 32 x 32
    l = batch_norm(ConvLayer(l_in, num_filters=16, filter_size=(3,3), stride=(1,1), nonlinearity=rectify, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
        
    # first stack of residual blocks, output is 16 x 32 x 32
    for _ in range(n):
        l = residual_block(l)
    
    # second stack of residual blocks, output is 32 x 16 x 16
    l = residual_block(l, increase_dim=True)
    for _ in range(1,n):
        l = residual_block(l)
    
    # third stack of residual blocks, output is 64 x 8 x 8
    l = residual_block(l, increase_dim=True)
    for _ in range(1,n-1):
        l = residual_block(l) 
    
    l = residual_block(l,last=True)
    # average pooling
    l = GlobalPoolLayer(l)
    # fully connected layer
    network = DenseLayer(
            l, num_units=100,
            W=lasagne.init.HeNormal(),
            nonlinearity=lasagne.nonlinearities.sigmoid)
     
    return network,l

############################## Batch iterator ###############################

def iterate_minibatches(inputs, targets, batchsize, shuffle=False, augment=False):
    
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        if augment:
            # as in paper : 
            # pad feature arrays with 4 pixels on each side
            # and do random cropping of 32x32
            padded = np.pad(inputs[excerpt],((0,0),(0,0),(4,4),(4,4)),mode='constant')
            random_cropped = np.zeros(inputs[excerpt].shape, dtype=np.float32)
            crops = np.random.random_integers(0,high=8,size=(batchsize,2))
            for r in range(batchsize):
                # Cropping and possible flipping
                if (np.random.randint(2) > 0):
                    random_cropped[r,:,:,:] = padded[r,:,crops[r,0]:(crops[r,0]+32),crops[r,1]:(crops[r,1]+32)]
                else:
                    random_cropped[r,:,:,:] = padded[r,:,crops[r,0]:(crops[r,0]+32),crops[r,1]:(crops[r,1]+32)][:,:,::-1]
            inp_exc = random_cropped
        else:
            inp_exc = inputs[excerpt]
        
        yield inp_exc, targets[excerpt]

def iterate_minibatches_pair(inputs, targets, inputs_pair, targets_pair, batchsize, shuffle=False, augment=False):
    assert len(inputs) == len(targets)
    assert len(inputs) == len(inputs_pair)
    assert len(inputs_pair) == len(targets_pair)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        if augment:
            # as in paper :
            # pad feature arrays with 4 pixels on each side
            # and do random cropping of 32x32
            padded = np.pad(inputs[excerpt],((0,0),(0,0),(4,4),(4,4)),mode='constant')
            padded_pair = np.pad(inputs_pair[excerpt],((0,0),(0,0),(4,4),(4,4)),mode='constant')
            random_cropped = np.zeros(inputs[excerpt].shape, dtype=np.float32)
            random_cropped_pair = np.zeros(inputs_pair[excerpt].shape, dtype=np.float32)
            crops = np.random.random_integers(0,high=8,size=(batchsize,2))
            for r in range(batchsize):
                # Cropping and possible flipping
                if (np.random.randint(2) > 0):
                    random_cropped[r,:,:,:] = padded[r,:,crops[r,0]:(crops[r,0]+32),crops[r,1]:(crops[r,1]+32)]
                    random_cropped_pair[r,:,:,:] = padded_pair[r,:,crops[r,0]:(crops[r,0]+32),crops[r,1]:(crops[r,1]+32)]
                else:
                    random_cropped[r,:,:,:] = padded[r,:,crops[r,0]:(crops[r,0]+32),crops[r,1]:(crops[r,1]+32)][:,:,::-1]
                    random_cropped_pair[r,:,:,:] = padded_pair[r,:,crops[r,0]:(crops[r,0]+32),crops[r,1]:(crops[r,1]+32)][:,:,::-1]
            inp_exc = np.append(random_cropped, random_cropped_pair, axis = 0)
        else:
            inp_exc = np.append(inputs[excerpt], inputs_pair[excerpt], axis = 0)
        yield inp_exc, np.append(targets[excerpt], targets_pair[excerpt], axis = 0)


def accuracy_measure(X_valid, Y_valid, class_means, val_fn, top1_acc_list, iteration, iteration_total, type_data):
    
    stat_hb1   = []
    stat_icarl = []
    stat_ncm   = []
    
    for batch in iterate_minibatches(X_valid, Y_valid, min(500,len(X_valid)), shuffle=False):
        inputs, targets_prep = batch
        targets = np.zeros((inputs.shape[0],100),np.float32)
        targets[range(len(targets_prep)),targets_prep.astype('int32')] = 1.
        err,pred,pred_inter = val_fn(inputs, targets)
        pred_inter  = (pred_inter.T/np.linalg.norm(pred_inter.T,axis=0)).T
        
        # Compute score for iCaRL
        sqd         = cdist(class_means[:,:,0].T, pred_inter, 'sqeuclidean')                    
        score_icarl = (-sqd).T
        # Compute score for NCM
        sqd         = cdist(class_means[:,:,1].T, pred_inter, 'sqeuclidean')                    
        score_ncm   = (-sqd).T
        
        # Compute the accuracy over the batch
        stat_hb1   += ([ll in best for ll, best in zip(targets_prep.astype('int32'), np.argsort(pred, axis=1)[:, -1:])])
        stat_icarl += ([ll in best for ll, best in zip(targets_prep.astype('int32'), np.argsort(score_icarl, axis=1)[:, -1:])])
        stat_ncm   += ([ll in best for ll, best in zip(targets_prep.astype('int32'), np.argsort(score_ncm, axis=1)[:, -1:])])
    
    print("Final results on "+type_data+" classes:")
    print("  top 1 accuracy iCaRL          :\t\t{:.2f} %".format(np.average(stat_icarl)* 100))
    print("  top 1 accuracy Hybrid 1       :\t\t{:.2f} %".format(np.average(stat_hb1)* 100))
    print("  top 1 accuracy NCM            :\t\t{:.2f} %".format(np.average(stat_ncm)* 100))
    
    top1_acc_list[iteration,0,iteration_total] = np.average(stat_icarl) * 100
    top1_acc_list[iteration,1,iteration_total] = np.average(stat_hb1) * 100
    top1_acc_list[iteration,2,iteration_total] = np.average(stat_ncm) * 100
    
    return top1_acc_list
