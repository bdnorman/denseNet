import tensorflow as tf
import sys
sys.path.insert(0,'../')
import general_functions as gf

def dense_net0(input_tensor, age, sex, race, NUM_OF_CLASSES, BATCH_SIZE, growth_rate, batch_norm=False, include_dems = False):
    '''

    :param input_tensor: image tensor of shape [BATCH, HEIGHT, WIDTH, CHANNELS]
    :param age: age tensor of shape [BATCH, 1]
    :param sex: sex tensor of shape [BATCH, 1]
    :param race: race tensor of shape [BATCH, 1]
    :param NUM_OF_CLASSES: number of classifcation groups to make
    :param BATCH_SIZE: batch size being trained with
    :param growth_rate: number of channels to use in dense blocks
    :param batch_norm: boolean on whether or not to use batch normalization after convolutions. Default is false.
    :param include_dems: boolean on whether or not to include inputted demographic vectore. Default is false.
    :return: results and logits from model, where results is just the argmax across that channel dimension of the logits.
    '''
    demographic_input = tf.concat([age, sex, race], 1)
    conv1 = gf.conv_layer2d(input_tensor, [7, 7, 1, 16], 16, name="conv1")
    x = gf.maxpool_layer2d(conv1,pool_size=[1,3,3,1],stride_size=[1,2,2,1], name="pool1")

    #Dense Block 1
    for b in range(6):
        in_channel = x.get_shape().as_list()[3]
        xn=gf.composite_fun(x,[3, 3], growth_rate, bn=batch_norm, padding='SAME', name="dense1_conv"+str(b))
        x = tf.concat([x,xn], 3)
    #Transition Layer 1
    in_channel = x.get_shape().as_list()[3]
    x1 = gf.conv_layer2d(x, [1, 1, in_channel, in_channel], in_channel, name="trans1")
    x1 = gf.avgpool_layer2d(x1, pool_size=[1,2,2,1], stride_size=[1,2,2,1], name="pool2")
    #Dense Block 2
    for b in range(6):
        in_channel = x1.get_shape().as_list()[3]
        xn=gf.composite_fun(x1,[3, 3, in_channel,growth_rate], growth_rate, bn=batch_norm, name="dense2_conv"+str(b))
        x1 = tf.concat([x1,xn], 3)
    # Transition Layer 2
    in_channel = x1.get_shape().as_list()[3]
    x2 = gf.conv_layer2d(x1, [1, 1, in_channel, in_channel], in_channel, name="trans2")
    x2 = gf.avgpool_layer2d(x2,pool_size=[1,2,2,1],stride_size=[1,2,2,1], name="pool2")
    # Dense Block 3
    for b in range(6):
        in_channel = x2.get_shape().as_list()[3]
        xn = gf.composite_fun(x2, [3, 3, in_channel, growth_rate], growth_rate, bn=batch_norm, name="dense3_conv" + str(b))
        x2 = tf.concat([x2, xn], 3)

    # Transition Layer 3
    in_channel = x2.get_shape().as_list()[3]
    x3 = gf.conv_layer2d(x2, [1, 1, in_channel, in_channel], in_channel, name="trans3")
    x3 = gf.avgpool_layer2d(x3,pool_size=[1,2,2,1],stride_size=[1,2,2,1], name="pool3")
    # Dense Block 4
    for b in range(6):
        in_channel = x3.get_shape().as_list()[3]
        xn = gf.composite_fun(x3, [3, 3, in_channel, growth_rate], growth_rate, bn=batch_norm, name="dense4_conv" + str(b))
        x3 = tf.concat([x3, xn], 3)

    x_last = tf.nn.relu(x3)
    last_size = x_last.get_shape().as_list()
    # full_conv = gf.conv_layer2d(x_last, [last_size[1], last_size[2]], last_size[3], padding='VALID', name='full_conv')
    # with tf.name_scope('reshape'):
    #     full_conv_flat = tf.reshape(full_conv, [BATCH_SIZE, -1])
    # fc1 = tf.layers.dropout(tf.layers.dense(full_conv_flat, 200, activation=tf.nn.relu, name="fc1"), rate=0.5)

    last_pool = gf.avgpool_layer2d(x_last, pool_size=[1, last_size[1], last_size[2], 1], stride_size=[1, 1, 1, 1],padding='VALID', name="last_pool")

    #Demographic insert
    with tf.name_scope('reshape'):
        full_conv_flat = tf.reshape(last_pool, [BATCH_SIZE, -1])
    if include_dems:
        with tf.name_scope('demographic_add'):
            demographic_input = tf.concat([age, sex, race], 1)
            dem_dense = tf.to_float(
                tf.layers.dense(demographic_input, 32, activation=tf.nn.relu, name='dem_graphic_dense'))
            full_conv_flat = tf.concat([full_conv_flat, dem_dense], axis=1, name='dem_im_join')

    with tf.name_scope("logits"):
        logits = tf.layers.dense(full_conv_flat, NUM_OF_CLASSES, name='fc2')
        result = tf.argmax(logits, axis = 1)
    return result, logits


def alex_net_mini_1(input_tensor, age, sex, race, NUM_OF_CLASSES, BATCH_SIZE, batch_norm, prob, include_dems = False):
    '''

    :param input_tensor: image tensor of shape [BATCH, HEIGHT, WIDTH, CHANNELS]
    :param age: age tensor of shape [BATCH, 1]
    :param sex: sex tensor of shape [BATCH, 1]
    :param race: race tensor of shape [BATCH, 1]
    :param NUM_OF_CLASSES: number of classifcation groups to make
    :param BATCH_SIZE: batch size being trained with
    :param prob: dropout rate to keep. Defaulted to 0.5 in net_runner.py and to 1.0 in acc_class.py
    :param include_dems: boolean on whether or not to include inputted demographic vectore. Default is false.
    :return: results and logits from model, where results is just the argmax across that channel dimension of the logits.
    '''
    conv1 = gf.conv_layer2d(input_tensor, [5, 5, 1, 24], 24, bn=batch_norm, name="conv1")
    conv2 = gf.conv_layer2d(conv1, [5, 5, 24, 24], 24, bn=batch_norm, name="conv2")
    stacked_conv = tf.concat((conv1, conv2), 3, name = 'stacked_conv')
    pool1 = gf.maxpool_layer2d(stacked_conv, pool_size=[1,4,4,1], stride_size=[1,4,4,1], name="pool1")

    conv3 = gf.conv_layer2d(pool1, [3, 3, 48, 96], 96, bn=batch_norm, name="conv3")
    pool2 = gf.maxpool_layer2d(conv3, name="pool2")  # shape: [1, 9, 20, 11, 64]
    with tf.name_scope("reshape"):
        conv3_flat = tf.reshape(pool2, [BATCH_SIZE, pool2.get_shape().as_list()[1] * pool2.get_shape().as_list()[2] *
                                        pool2.get_shape().as_list()[3]], name = "flat_conv")
    if include_dems:
        with tf.name_scope('demographic_add'):
            demographic_input = tf.concat([age, sex, race], 1)
            dem_dense = tf.to_float(tf.layers.dense(demographic_input, 32, activation=tf.nn.relu, name='dem_graphic_dense'))
            conv3_flat = tf.concat([conv3_flat, dem_dense], axis=1, name='dem_im_join')
    dense1 = tf.nn.dropout(tf.layers.dense(conv3_flat, 400, activation=tf.nn.relu, name="fc1"), keep_prob=prob)
    with tf.name_scope("logits"):
        logits = tf.nn.softmax(tf.layers.dense(dense1, NUM_OF_CLASSES, name="fc3"))
        result = tf.argmax(logits, axis=1)
    return result, logits

def alex_net_mini_2(input_tensor, age, sex, race, NUM_OF_CLASSES, BATCH_SIZE, batch_norm, prob, include_dems=False):
    '''

    :param input_tensor: image tensor of shape [BATCH, HEIGHT, WIDTH, CHANNELS]
    :param age: age tensor of shape [BATCH, 1]
    :param sex: sex tensor of shape [BATCH, 1]
    :param race: race tensor of shape [BATCH, 1]
    :param NUM_OF_CLASSES: number of classifcation groups to make
    :param BATCH_SIZE: batch size being trained with
    :param prob: dropout rate to keep. Defaulted to 0.5 in net_runner.py and to 1.0 in acc_class.py
    :param include_dems: boolean on whether or not to include inputted demographic vectore. Default is false.
    :return: results and logits from model, where results is just the argmax across that channel dimension of the logits.
    '''

    conv1 = tf.nn.relu(gf.conv_layer2d(input_tensor, [3, 3, 1, 24], 24, bn=batch_norm, name="conv1", padding='VALID'))
    conv2 = tf.nn.relu(gf.conv_layer2d(conv1, [3, 3, 24, 24], 24, bn=batch_norm, name="conv2", padding='VALID'))
    conv3 = tf.nn.relu(gf.conv_layer2d(conv2, [3, 3, 24, 24], 24, bn=batch_norm, name="conv3", padding='VALID'))
    pool1 = gf.maxpool_layer2d(conv3, name="pool1")

    conv4 = tf.nn.relu(gf.conv_layer2d(pool1, [3, 3, 24, 24], 24, bn=batch_norm, name="conv4", padding='VALID'))
    conv5 = tf.nn.relu(gf.conv_layer2d(conv4, [3, 3, 24, 24], 24, bn=batch_norm, name="conv5", padding='VALID'))
    conv6 = tf.nn.relu(gf.conv_layer2d(conv5, [3, 3, 24, 24], 24, bn=batch_norm, name="conv6", padding='VALID'))
    pool2 = gf.maxpool_layer2d(conv6, name="pool2")  # shape: [1, 9, 20, 11, 64]
    with tf.name_scope("reshape"):
        conv3_flat = tf.reshape(pool2, [BATCH_SIZE, pool2.get_shape().as_list()[1] * pool2.get_shape().as_list()[2] *
                                        pool2.get_shape().as_list()[3]], name = "flat_conv")
    if include_dems:
        with tf.name_scope('demographic_add'):
            demographic_input = tf.concat([age, sex, race], 1)
            dem_dense = tf.to_float(tf.layers.dense(demographic_input, 32, activation=tf.nn.relu, name='dem_graphic_dense'))
            conv3_flat = tf.concat([conv3_flat, dem_dense], axis=1, name='dem_im_join')
    dense1 = tf.nn.dropout(tf.layers.dense(conv3_flat, 200, activation=tf.nn.relu, name="fc1"), keep_prob=prob)
    with tf.name_scope("logits"):
        logits = tf.nn.softmax(tf.layers.dense(dense1, NUM_OF_CLASSES, name="fc3"))
        result = tf.argmax(logits, axis=1)
    return result, logits



