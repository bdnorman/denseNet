import tensorflow as tf
import sys

def input_pipeline(file_paths, BATCH_SIZE, IMAGE_SIZE, class_type=''):
    '''

    :param file_paths: tensor string constant of training files list
    :param BATCH_SIZE: batch size to use for training
    :param IMAGE_SIZE: height, width depth integers of image
    :param class_type: classification type to make for kl grading. Defaults to original 0-4 classification. Options are
    are `binary` (0-1 vs. 2-4), `three_class` (0-1 vs. 2-3 vs. 4), four_class (0-1 vs. 2 vs. 3 vs. 4)
    :return: image tensor, outpout tensor, age, tensor, sex tensor, race tensor
    '''
    filename_queue = tf.train.string_input_producer(file_paths, shuffle=True)
    height = IMAGE_SIZE[0]
    width = IMAGE_SIZE[1]
    depth = IMAGE_SIZE[2]
    channels = 1
    in_type = tf.float32
    image_size = height * width * depth
    image_bytes_in = ((height * width * depth * channels)+4) * in_type.size
    binary_class=False
    three_class=False
    four_class = False
    if class_type=='binary':
        binary_class=True
    elif class_type=='three_class':
        three_class=True
    elif class_type=='four_class':
        four_class = True
    elif class_type!='':
        sys.exit('%s is not a valid class type. Must be `binary`, `three_class`, or empty.' % class_type)
    reader_in = tf.FixedLengthRecordReader(record_bytes=image_bytes_in)
    key, value = reader_in.read(filename_queue)
    record_bytes_in = tf.decode_raw(value, in_type)
    #long_im = tf.strided_slice(record_bytes_in, [0], [image_size])
    image = tf.reshape(tf.strided_slice(record_bytes_in, [4], [image_size+4]),
                             [width, height, depth])
    #Only use this transpose statement if image were created with 'F' style flattening
    #image = tf.transpose(image, [2,1 ,0])
    image = tf.to_float(image)
    image = tf.expand_dims(image, axis = 0)

    image = tf.div(tf.subtract(image, tf.reduce_min(image)), tf.subtract(tf.reduce_max(image), tf.reduce_min(image)))
    #image = tf.nn.l2_normalize(image, dim=0)

    kl_score = tf.reshape(tf.strided_slice(record_bytes_in,[0], [1]), [1])
    kl_score = tf.to_int32(kl_score)
    kl_score = tf.expand_dims(kl_score, axis = 0)

    sex = tf.reshape(tf.strided_slice(record_bytes_in,[1], [2]), [1])
    sex = tf.to_int32(sex)
    sex = tf.expand_dims(sex, axis = 0)

    age = tf.reshape(tf.strided_slice(record_bytes_in,[2], [3]), [1])
    age = tf.to_int32(age)
    age = tf.expand_dims(age, axis = 0)

    race = tf.reshape(tf.strided_slice(record_bytes_in,[3], [4]), [1])
    race = tf.to_int32(race)
    race = tf.expand_dims(race, axis = 0)

    min_fraction_of_examples_in_queue = 0.5
    min_queue_examples = int(2 *
                             min_fraction_of_examples_in_queue)
    batch_input, batch_output, sex_batch, age_batch, race_batch = tf.train.shuffle_batch([image, kl_score, sex, age, race], batch_size=BATCH_SIZE, capacity=3 * BATCH_SIZE + min_queue_examples,
                                                       enqueue_many=True, min_after_dequeue=min_queue_examples, num_threads=16)


    #Make binary indexing conversion
    if binary_class:
        comp = tf.less_equal(batch_output, tf.to_int32(tf.ones_like(batch_output)))
        batch_output = tf.where(comp, tf.zeros_like(batch_output), tf.ones_like(batch_output))

    #Make 3 class indexing
    if three_class:
        comp1 = tf.less_equal(batch_output, tf.to_int32(tf.ones([BATCH_SIZE, 1])))
        ones_identifier = tf.where(comp1, tf.zeros_like(batch_output), tf.ones_like(batch_output))

        full_lesion_val = tf.constant(4)
        comp2 = tf.equal(batch_output, tf.scalar_mul(full_lesion_val, tf.to_int32(tf.ones_like(batch_output))))
        full_lesion_identifier = tf.where(comp2, tf.ones_like(batch_output), tf.zeros_like(batch_output))
        batch_output = tf.add(ones_identifier, full_lesion_identifier)
    #Make 4 class indexing
    if four_class:
        comp1 = tf.less_equal(batch_output, tf.to_int32(tf.ones([BATCH_SIZE, 1])))
        ones_identifier = tf.where(comp1, tf.zeros_like(batch_output), tf.ones_like(batch_output))

        mild_lesion_val = tf.constant(3)
        comp2 = tf.greater_equal(batch_output, tf.scalar_mul(mild_lesion_val, tf.to_int32(tf.ones_like(batch_output))))
        mild_lesion_identifier = tf.where(comp2, tf.ones_like(batch_output), tf.zeros_like(batch_output))

        severe_lesion_val = tf.constant(4)
        comp3 = tf.greater_equal(batch_output, tf.scalar_mul(severe_lesion_val, tf.to_int32(tf.ones_like(batch_output))))
        severe_lesion_identifier = tf.where(comp3, tf.ones_like(batch_output), tf.zeros_like(batch_output))

        batch_output = tf.add(ones_identifier, tf.add(mild_lesion_identifier, severe_lesion_identifier))


    batch_output = tf.squeeze(batch_output, axis=-1)
    # sex_batch = tf.squeeze(sex_batch, axis=-1)
    # age_batch = tf.squeeze(age_batch, axis=-1)
    # race_batch = tf.squeeze(race_batch, axis=-1)
    return batch_input, batch_output, sex_batch, age_batch, race_batch










