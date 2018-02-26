import tensorflow as tf
import os
from os.path import join
from architecture_helper import *
import queue_input as data_read
import time
import argparse
import acc_class as acc
import sys
sys.path.insert(0,'../')
import general_functions as gf
import pickle

class networkBuilder():
    '''
    class that allows for building the required network (save folders, tensors, architecutre)
    and running the built network.
    '''
    def __init__(self, args, IMAGE_SIZE, BATCH_SIZE, EPOCHS):
        '''

        :param args: arparse inputs
        :param IMAGE_SIZE: height, width, depth or .raw image to be loaded
        :param BATCH_SIZE: batch size to train with, hard coded to 30
        :param EPOCHS: number of epochs to run model for
        '''
        self.args=args
        self.IMAGE_SIZE = IMAGE_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS

    def build_network(self, train_im_path, valid_im_path = None, test_im_path = None, train_aug_im_path = None):
        '''

        :param train_im_path: path directory to .raw training files. This is required
        :param valid_im_path: path directory to .raw validation files. This is not required. If
        argument not passed validation accuracy will not be calculated or saved.
        :param test_im_path: path directory to .raw testing files. This is not required. If
        argument not passed testing accuracy will not be calculated or saved.
        :param train_aug_im_path: path directory to .raw augmented training files. This is not required. If
        argument not only the train_im_path file will be used training/epoch calculation.
        :return: nothing physically returned, tensor variables, tf session, and saving directories are saved
        as class variables
        '''
        self.train_im_path = train_im_path
        self.valid_im_path = valid_im_path
        self.test_im_path = test_im_path
        self.train_aug_im_path = train_aug_im_path

        GPU_VIS = self.args.gpus_vis

        project = self.args.modelname
        self.version = self.args.modeltype + ('_dems' if self.args.dem_include else '') + ('_'+ self.args.class_type if self.args.class_type else '') + ('_batchnorm' if self.args.batch_norm else '')
        self.model_name = project + '_' + self.version

        #Reload parameters from old model
        if self.args.model_params:
            previous_param_dict = pickle.load(open(self.args.model_params, 'rb'))
            model =previous_param_dict['model']
            WEIGHTS = previous_param_dict['WEIGHTS']
            class_type = previous_param_dict['class_type']
            loss_type = previous_param_dict['loss_type']
            restore_model = previous_param_dict['restore_model']
            use_batchn = previous_param_dict['use_batchn']
            use_dems = previous_param_dict['use_dems']

        #Overwrite any specified parameters loaded by model params OR just specify them initially
        if self.args.model_weights:
            WEIGHTS = self.args.model_weights
        if self.args.modeltype:
            model = self.args.modeltype

        #Manually set defaults so reloaded parameters aren't overwritten
        if self.args.class_type:
            class_type = self.args.class_type
        elif not self.args.class_type and not self.args.model_params:
            class_type = ''
        if self.args.loss:
            loss_type = self.args.loss
        elif not self.args.loss and not self.args.model_params:
            loss_type = 'cross_entropy'
        if self.args.restore_model:
            restore_model = self.args.restore_model
        elif not self.args.restore_model and not self.args.model_params:
            restore_model = ''
        if self.args.batch_norm:
            use_batchn = self.args.batch_norm
        elif not self.args.batch_norm and not self.args.model_params:
            use_batchn = False
        if self.args.dem_include:
            use_dems = self.args.dem_include
        elif not self.args.dem_include and not self.args.model_params:
            use_dems = False
        #Save all new model params into dictionary
        new_param_dict = {}
        new_param_dict['model'] = model
        new_param_dict['WEIGHTS'] = WEIGHTS
        new_param_dict['class_type'] = class_type
        new_param_dict['loss_type'] = loss_type
        new_param_dict['restore_model'] = restore_model
        new_param_dict['use_batchn'] = use_batchn
        new_param_dict['use_dems'] = use_dems

        NUM_OF_CLASSES = len(WEIGHTS)

        doc_folder = '~/densenet/Doc/'

        logs_dir = join(doc_folder, project, self.version,'models')
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        self.logs_dir = logs_dir

        output_log = join(doc_folder, project, self.version, 'logs')
        if not os.path.exists(output_log):
            os.makedirs(output_log)
        #write dictionary of parameter values to logs folder
        pickle.dump(new_param_dict, open(join(output_log,'param_dict.p'), 'wb'))


        number_log_files = len([name for name in os.listdir(output_log) if 'train' in name])
        file_ver = number_log_files + 1
        new_log = join(output_log, 'train_log' + str(file_ver) + '.txt')

        #Prompt comes up if a log file already exists for the given model and version type
        if number_log_files > 0:
            old_log = join(output_log, 'train_log' + str(number_log_files) + '.txt')
            dec = raw_input(
                'Log file for these model name and version already exists indicating that this model may have been run before. Do you wish to continue? [y (yes) /n (no)/ d (delete old file)]:')
            if dec == 'n':
                sys.exit('User chose to terminate run since this model appears to have been run before')
            elif dec == 'd':
                #Deletes the most recent log file
                os.remove(old_log)
                new_log = old_log
                file_ver -= 1
        self.file_ver = file_ver
        self.new_log = new_log
        self.accuracy_funs = acc.acc_calc(self.IMAGE_SIZE, self.BATCH_SIZE, class_type=class_type)

        train_path = gf.create_raw_file_list(train_im_path)
        if train_aug_im_path:
            train_aug_path = gf.create_raw_file_list(train_aug_im_path)
            train_path_files = train_path+train_aug_path
        else:
            train_path_files = train_path
        self.train_path_files = train_path_files
        train_path_t = tf.constant(self.train_path_files, dtype=tf.string)

        self.x, self.y, self.sex, self.age, self.race = data_read.input_pipeline(train_path_t, self.BATCH_SIZE, self.IMAGE_SIZE, class_type=class_type)
        y_exp = tf.one_hot(self.y, depth=NUM_OF_CLASSES, axis=-1)
        prob = tf.placeholder_with_default(0.5, shape=())
        self.result, self.logits = model_selector(model, self.x, self.age, self.sex, self.race, NUM_OF_CLASSES, self.BATCH_SIZE,growth_rate=12,
                                        batch_normalization=use_batchn, prob=prob, include_dems=use_dems)
        with tf.name_scope("loss1"):
            y_exp_vec = tf.reshape(y_exp, [-1, NUM_OF_CLASSES])
            pos_weights1 = tf.constant([WEIGHTS], dtype='float32')
            weight_per_lab1 = tf.transpose(tf.matmul(y_exp_vec, tf.transpose(pos_weights1)))
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits)
            cross_entropy = tf.reshape(cross_entropy, [-1])
            cross_entropy_1 = tf.multiply(weight_per_lab1, cross_entropy)
            loss1 = tf.reduce_mean(cross_entropy_1)
            tf.summary.scalar('loss', loss1)
        with tf.name_scope("loss2"):
            loss2 = tf.losses.mean_squared_error(self.y, self.result)
        with tf.name_scope("loss3"):
            loss3 = tf.losses.mean_squared_error(self.y, self.result)

        if loss_type == 'cross_entropy':
            final_loss = loss1
        elif loss_type == 'mse':
            final_loss = loss2
        elif loss_type =='mse_ce_comb':
            loss1_weight = tf.Variable(tf.constant([0.5]), name='loss1_weight', dtype=tf.float32)
            loss2_weight = tf.subtract(tf.constant([1.0]), loss1_weight)
            final_loss = tf.div(tf.add(tf.multiply(loss1, loss1_weight), tf.multiply(loss2, loss2_weight)), tf.constant([2.0]))
        self.final_loss = final_loss

        self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss=final_loss, global_step=self.global_step)

        # Create session
        c = tf.ConfigProto()
        c.gpu_options.visible_device_list = GPU_VIS
        c.gpu_options.per_process_gpu_memory_fraction = self.args.gpu_usage
        c.allow_soft_placement = True
        c.log_device_placement = False
        self.sess = tf.Session(config=c)
        tf.global_variables_initializer().run(session=self.sess)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=30)

        sys.stdout = gf.Logger(new_log)
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print('%d total number of parameters in model' % total_parameters)


    def run_network(self):
        '''
        :return: This function actually runs the graph built in build_network for 20 epochs with BATCH_SIZE
        of 30. Also saves accuracy statistics on training, validation and testing (if supplied) datasets
        evory epoch. .ckpt model for graph weights are also saved every epoch.
        '''
        # Start input enqueue threads
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

        print(self.args)
        print('Data path: %s' % self.train_im_path)
        print('Training begining:')
        itrs_per_epoch = int(round(len(self.train_path_files)/self.BATCH_SIZE))
        print('Number of iterations per epoch: %d' % itrs_per_epoch)
        start_time = time.time()
        for epoch in range(self.EPOCHS):
            for itr in range(itrs_per_epoch):
                _, step = self.sess.run([self.train_op, self.global_step])
            result_im, input_im, loss = self.sess.run([self.result, self.y, self.final_loss])
            print("-----%s seconds between for epoch %d iterations-----" % (time.time() - start_time, epoch))
            start_time = time.time()
            print("Loss at epoch %d for model %s: %g" % (epoch, self.model_name, loss))
            print('Truth:')
            print(input_im)
            print('Predictions:')
            print(result_im)

            if epoch % 5 ==0:
                print('Training accuracy')
                total_labels, total_preds, total_logits = \
                    self.accuracy_funs.confmat_gen(self.train_im_path, self.sess, self.x, tf.squeeze(self.sex, axis = -1),tf.squeeze(self.age, axis = -1), tf.squeeze(self.race, axis = -1), self.result, self.logits)
                self.accuracy_funs.save_results('train',epoch, total_preds, total_labels, total_logits,
                                           self.args.modelname, self.version, self.new_log,
                                           self.file_ver
                                           )
                if self.valid_im_path:
                    print('Validation accuracy')
                    total_labels, total_preds, total_logits = \
                        self.accuracy_funs.confmat_gen(self.valid_im_path, self.sess, self.x, tf.squeeze(self.sex, axis = -1),
                                                  tf.squeeze(self.age, axis = -1), tf.squeeze(self.race, axis = -1), self.result, self.logits)
                    self.accuracy_funs.save_results('valid',epoch, total_preds, total_labels, total_logits,
                                               self.args.modelname, self.version, self.new_log,
                                               self.file_ver
                                               )

                if self.test_im_path:
                    print('Testing accuracy')
                    total_labels, total_preds, total_logits = \
                        self.accuracy_funs.confmat_gen(self.test_im_path, self.sess, self.x, tf.squeeze(self.sex, axis = -1),
                                                  tf.squeeze(self.age, axis = -1), tf.squeeze(self.race, axis = -1), self.result, self.logits)
                    self.accuracy_funs.save_results('test',epoch, total_preds, total_labels, total_logits,
                                               self.args.modelname, self.version, self.new_log,
                                               self.file_ver
                                               )
                #Save model
                if epoch>0:
                    checkpoint_file = join(self.logs_dir, "check_point" + str(epoch))
                    if not os.path.exists(checkpoint_file):
                        os.makedirs(checkpoint_file)
                    self.saver.save(self.sess, checkpoint_file + "/" + self.model_name + ".ckpt")

def main():
    '''
    Main function the allows for parameter setting. Calls networkBuilder's build_networks() and
    run_network() functions with the argparse inputs and a BATCH_SIZE of 30 for 20 EPOCHS.
    :return: model checkpoints, outputted console screen .txt file, model parameters pickle file, and
    model stats per epoch pickle file (see README.md 'Outputs' section)
    '''
    parser = argparse.ArgumentParser(description='Run convoluitonal neural network for KL grade detection')

    parser.add_argument('--gpus_vis', nargs='?', metavar='gpus_visible', type=str,
                        help='GPU number that model can see. Default is 0.', default='0')
    parser.add_argument('--gpu_usage', nargs='?', type=float, help='Percentage of GPU to use. Default is 1.0.',
                        default=1.0)
    parser.add_argument('--modelname', type=str, help='General name of the model', required=True)
    parser.add_argument('--modeltype', type=str,
                        help='Name of model architecture. Is not required unless previous run parameters are not loaded.')
    parser.add_argument('--model_params', type=str, help='File path to previous model pickle file.', required=False)

    # Optional model parameters
    parser.add_argument('--image_size', nargs='+', type=int, help = 'Height width and depth of inputted images.')
    parser.add_argument('--model_weights', nargs='+', type=float, help='Weights of model classes.', required=False)
    parser.add_argument('--class_type', type=str, help='Number of classifications to make', required=False)
    parser.add_argument('--loss', type=str,
                        help='Loss funciton to use, can either be: `cross_entropy`, `mse`, `mse_ce_comb`. Default if cross_entropy.',
                        required=False)
    parser.add_argument('--restore_model', type=str, help='Full path of restore model', required=False)
    parser.add_argument('--batch_norm', type=gf.str2bool, help='Batch norm boolean', required=False)
    parser.add_argument('--dem_include', type=gf.str2bool,
                        help='Boolean on whether or not to include demographic information in model architecture',
                        required=False)

    args = parser.parse_args()

    train_im_path = ''
    train_aug_im_path = ''
    valid_im_path = ''
    test_im_path = ''

    IMAGE_SIZE = args.image_size
    BATCH_SIZE = 30
    EPOCH = 200

    net_build = networkBuilder(args, IMAGE_SIZE, BATCH_SIZE, EPOCH)
    net_build.build_network(train_im_path, valid_im_path, test_im_path, train_aug_im_path)
    net_build.run_network()

if __name__ == '__main__':
    main()