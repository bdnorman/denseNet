#general functions used throughout coding
from __future__ import division
import numpy as np
import code
import os
from os.path import join
from os.path import isfile
import sys
import scipy.io as sio
import tensorflow as tf
import dicom
from dicom.errors import InvalidDicomError
import argparse
import sklearn.metrics as smk


### GENERAL HELPER FUNCTIONS ###
def keyboard(banner=None):
    ''' Function that mimics the matlab keyboard command '''
    # use exception trick to pick up the current frame
    try:
        raise None
    except:
        frame = sys.exc_info()[2].tb_frame.f_back
    print("# Use quit() to exit :) Happy debugging!")
    # evaluate commands in current namespace
    namespace = frame.f_globals.copy()
    namespace.update(frame.f_locals)
    try:
        code.interact(banner=banner, local=namespace)
    except SystemExit:
        return

def getImageFromRaw_adv(fileName, IMAGE_SIZE, readorder = 'F', data_type = np.int16):
    '''

    :param fileName: full file path to .raw file
    :param IMAGE_SIZE: array of height, width, and depth of .raw file
    :param readorder: order in which the input images were flattened (default is to 'F' meaning it was written in matlab,
    'C' if image was flattened using anything else, mainly python)
    :param data_type: data type that .raw file was saved as. Default is int16
    :return: image array, worms score, sex, age, and race values
    '''
    height=IMAGE_SIZE[0]
    width=IMAGE_SIZE[1]
    depth=IMAGE_SIZE[2]
    raw_array = np.fromfile(fileName, dtype=data_type)
    worms = raw_array[0]
    sex = raw_array[1]
    age = raw_array[2]
    race = raw_array[3]
    image = np.reshape(raw_array[4:], (height, width, depth), order=readorder)
    return image, worms, sex, age, race

def array2gray(im):
    '''

    :param im: numpy array
    :return: 0-1 float 32 normalized array
    '''
    num = im-np.min(im)
    num = num.astype(float)
    denom = np.max(im)-np.min(im)
    denom = denom.astype(denom)
    return np.float32(num/denom)

def tf_array2gray(image):
    '''

    :param image: tensor array
    :return: 0-1 normalized tensor array
    '''
    return tf.div(tf.subtract(image, tf.reduce_min(image)), tf.subtract(tf.reduce_max(image), tf.reduce_min(image)))


def transform(filename):
    '''

    :param filename: full file path to .mat file
    :return: returns first array in .mat file
    '''
    file = sio.loadmat(filename)
    for key in file:
        if isinstance(file[key], np.ndarray):
            img = file[key]
    return img

def create_raw_file_list(data_path):
    '''

    :param data_path: directory path
    :return: list of full file directory path within data_path folder
    '''
    onlyfiles = [data_path + f for f in os.listdir(data_path) if isfile(join(data_path, f))]
    return onlyfiles


def str2bool(v):
    '''

    :param v: input from argpase
    :return: True or False boolean
    '''
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def confmat_combine(cm, groupings=None):
    '''
    Groups adjacent classes in a confusion matrix and recalcuates accuracies
    :param cm: confusion matrix
    :param groupings: groupings of adjancent classes to make new confusion matrix. Must be given as list of lists
    i.e. groupings = [[0,1], [2], [3,4]], to create a new 3 class confusion matrix
    :return: new confusion matrix, sensitivities and specificites of confusion matrix
    '''
    if groupings is None:
        groupings = [[i] for i in range(cm.shape[0])]
    new_class_n = len(groupings)
    new_cm = np.zeros((new_class_n, new_class_n))
    off_set = 0
    for c1 in range(new_class_n):
        count = 0
        number_row_to_go_down = len(groupings[c1])
        for c in groupings:
            start = c[0]
            end = c[-1] + 1
            new_cm[c1, count] = np.sum(cm[off_set:off_set + number_row_to_go_down, start:end])
            count += 1
        off_set += number_row_to_go_down
    train_acc = np.diag(new_cm).astype(float) / np.sum(new_cm, axis=1).astype(float)
    new_cm = np.asarray(new_cm, dtype=np.int)

    # https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
    FP = np.sum(new_cm, axis=0) - np.diag(new_cm)
    FN = np.sum(new_cm ,axis=1) - np.diag(new_cm)
    TP = np.diag(new_cm)
    TN = np.sum(new_cm) - (FP + FN + TP)

    # Sensitivity, hit rate, recall, or true positive rate
    sensitivity = TP / (TP + FN)
    # Specificity or true negative rate
    specificity = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)

    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)

    return new_cm, sensitivity, specificity


def all_acc_print(result_dict, data_type='valid', groupings = None):
    '''

    :param result_dict: dictionary from model generated results
    :param data_type: results from datasubset to print, options are `train`, `test`, `valid`. Default is set to `valid`
    :param groupings: groupings to make for confmat_combine function
    :return: prints sensitivity from each saved iteration
    '''
    for i in result_dict[data_type]['results'].keys():
        print(i)
        cm = result_dict[data_type]['results'][i]['confmat']
        new_cm, sensitivity,_ = confmat_combine(cm, groupings)
        print(sensitivity)

def results_gen(total_labels, total_preds, groupings=None):
    '''

    :param total_labels: truth labels as list
    :param total_preds: predicted labels as list
    :param groupings: groupings to make for confmat_combine function
    :return: prints confusion matrix, sensitivity, and specificity
    '''
    confmat = smk.confusion_matrix(total_labels, total_preds)
    cm, sensitivity, specificity = confmat_combine(confmat, groupings)
    print('Confusion matrix:')
    print(cm)
    print('Sensitivity:')
    print(sensitivity)
    print('Specificity')
    print(specificity)
    return cm, sensitivity, specificity

def parse_dicom_file(filename):
    """Parse the given DICOM filename
    :param filename: filepath to the DICOM file to parse
    :return: dictionary with DICOM image data
    """

    try:
        dcm = dicom.read_file(filename)
        dcm_image = dcm.pixel_array

        # try:
        #     intercept = dcm.RescaleIntercept
        # except AttributeError:
        #     intercept = 0.0
        # try:
        #     slope = dcm.RescaleSlope
        # except AttributeError:
        #     slope = 0.0
        #
        # if intercept != 0.0 and slope != 0.0:
        #     dcm_image = dcm_image * slope + intercept
        dcm_dict = {'pixel_data': dcm_image}
        return dcm_dict
    except InvalidDicomError:
        return None

def sort_results(base_dict, target_dict, itt, data_type = 'valid'):
    '''
    Function to sort results from one odel with respect to another so that they may be combined for ensembling
    base_dict: dictionary of the model that other results will be sorted to

    target_dict: dictionary of model that wants to be ensembled
    itt: iteration of model to be used from target_dict
    '''
    base_files = base_dict[data_type]['files']
    target_files = target_dict[data_type]['files']
    target_results = target_dict[data_type]['results'][str(itt)]['logits']
    target_info = zip(target_files, target_results)
    target_info.sort(key=lambda x: base_files.index(x[0]))
    sorted_results = [b for a, b in target_info]
    return sorted_results

def ensemble_results(truth_values, *arg):
    '''
    Combine desired logit results for however many models you want by averaging them

    truth_values: true values to predict
    *arg: whatever model logits you want to ensemble
    '''
    ens_results = []
    pred_length = len(truth_values)
    for i in range(pred_length):
        comb_result = np.argmax(np.mean([softmax(m[i]) for m in arg], axis=0))
        ens_results.append(comb_result)
    cm = smk.confusion_matrix(truth_values, ens_results)
    return cm


class Logger(object):
    # Set up logging
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

### FUNCTIONS FOR 2D NNs ###
# FLAGS = tf.app.flags.FLAGS
# tf.app.flags.DEFINE_boolean('use_fp16', False,
#                             """Train the model using fp16.""")

def conv_layer2d(x, convolution_size, output_channel, bn=False, padding='SAME', name='name'):
    '''
    Perform 2D convolution on inputted tensor
    :param x: image input tensor of size [BATCH, HEIGHT, WIDTH, CHANNELS]
    :param convolution_size: size of convlving window
    :param output_channel: size of channels in outpetted layer
    :param bn: boolean indicating whether or not to use batch normalization. Default is set to false.
    :param padding: whether or not to use 'SAME' of 'VALID' padding. Default is 'SAME'.
    :param name: name of convolution tensor.
    :return: outputted layer as tensor
    '''
    with tf.variable_scope(name) as scope:
        if bn:
            x = tf.contrib.layers.batch_norm(x, data_format='NHWC', scale=True, is_training=True)
        W_shape = [convolution_size[0], convolution_size[1], x.get_shape().as_list()[3], output_channel]
        b_shape = output_channel
        W = _variable_with_weight_decay(W_shape, 'W', wd=0.0)
        b = _variable_on_cpu(b_shape, 'B', tf.constant_initializer(0.1))
        tf.summary.histogram("weights", W)
        tf.summary.histogram("biases", b)
        return tf.nn.bias_add(tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding),b, name = scope.name)

def composite_fun(x,convoluiton_size, output_channel, bn=False, padding='SAME', name='name'):
    '''
    composit function used in dense net
    :param x: image input tensor of size [BATCH, HEIGHT, WIDTH, CHANNELS]
    :param convolution_size: size of convlving window
    :param output_channel: size of channels in outpetted layer
    :param bn: boolean indicating whether or not to use batch normalization. Default is set to false.
    :param padding: whether or not to use 'SAME' of 'VALID' padding. Default is 'SAME'.
    :param name: name of convolution tensor.
    :return: outputted layer as tensor
    '''
    if bn:
        x = tf.contrib.layers.batch_norm(x, data_format='NHWC', scale=True, is_training=True)
    x = tf.nn.relu(x)
    x = conv_layer2d(x, convoluiton_size, output_channel, bn=False, padding = padding, name = name)
    return x


def maxpool_layer2d(x, pool_size=[1,2,2,1], stride_size=[1,2,2,1], padding='SAME', name = "pool",):
    '''
    max pooling function
    :param x: image input tensor of size [BATCH, HEIGHT, WIDTH, CHANNELS]
    :param pool_size: size of average pooling to make of size [BATCH, HEIGHT, WIDTH, CHANNELS]. Default is [1,2,2,1]
    :param stride_size: stride size of pooling of size [BATCH, HEIGHT, WIDTH, CHANNELS]. Default is [1,2,2,1]
    :param padding: whether or not to use 'SAME' of 'VALID' padding. Default is 'SAME'.
    :param name: name of convolution tensor.
    :return: outputted layer as tensor
    '''
    with tf.name_scope(name):
        return tf.nn.max_pool(x, ksize=pool_size, strides=stride_size, padding=padding)

def avgpool_layer2d(x, pool_size=[1,2,2,1], stride_size=[1,2,2,1], padding='SAME',name="pool"):
    '''
    average pooling functions
    :param x: image input tensor of size [BATCH, HEIGHT, WIDTH, CHANNELS]
    :param pool_size: size of average pooling to make of size [BATCH, HEIGHT, WIDTH, CHANNELS]. Default is [1,2,2,1]
    :param stride_size: stride size of pooling of size [BATCH, HEIGHT, WIDTH, CHANNELS]. Default is [1,2,2,1]
    :param padding: whether or not to use 'SAME' of 'VALID' padding. Default is 'SAME'.
    :param name: name of convolution tensor.
    :return: outputted layer as tensor
    '''
    with tf.name_scope(name):
        return tf.nn.avg_pool(x, ksize=pool_size, strides=stride_size, padding=padding)



def _variable_on_cpu(shape, name, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    #dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    dtype = tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var

def _variable_with_weight_decay(shape, name, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  #dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  dtype = tf.float32
  var = _variable_on_cpu(
      shape,
      name,
      tf.truncated_normal_initializer(stddev=np.sqrt(2 / shape[2]), dtype=dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)