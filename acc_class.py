from __future__ import division
import numpy as np
import sys
import os
import scipy.io as sio
from os.path import isfile, join
import sklearn.metrics as skm
import pickle
sys.path.insert(0, '../')
import general_functions as gf

class acc_calc:
    call_number = 0
    def __init__(self, IMAGE_SIZE, BATCH_SIZE, class_type):
        self.IMAGE_SIZE = IMAGE_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.class_type = class_type


    def batch_set_gen(self, file_list):
        '''
        
        :param file_list: list of files
        :return: list of lists where each sublist contains BATCH_SIZE # of files. If the files_list is not
        evenly divisible by batch size then the last sublist is the remainder number of + (BATCH_SIZE - number of
        remaining files) from the begining of the file list 
        '''
        batch_rem = len(file_list) % self.BATCH_SIZE
        batch_sets = []
        for itr in range(0, len(file_list) - batch_rem, self.BATCH_SIZE):
            batch_sets.append(file_list[itr:(itr + self.BATCH_SIZE)])
        additional_records = file_list[0:self.BATCH_SIZE - batch_rem]
        remaining_vols = file_list[(len(file_list) - batch_rem):len(file_list)]
        add_records = remaining_vols + additional_records
        batch_sets.append(add_records)
        return batch_sets

    def getImageFromRaw(self, fileName):
        '''
        
        :param fileName: .raw file name
        :return: 0-1 normalized image array, prediction labels, sex, age, race
        '''
        height=self.IMAGE_SIZE[0]
        width=self.IMAGE_SIZE[1]
        depth=self.IMAGE_SIZE[2]
        raw_array = np.fromfile(fileName, dtype=np.float32)
        worms = raw_array[0]
        sex = raw_array[1]
        age = raw_array[2]
        race = raw_array[3]
        image = np.reshape(raw_array[4:], (height, width, depth), order='C')
        return gf.array2gray(image), worms, sex, age, race

    def confmat_gen(self, train_or_test_path, sess, x, sex, age, race, result, logits):
        '''
        
        :param train_or_test_path: directory path containing .raw files to be inputted into the model
        :param sess: tensorflow Session with an established graph
        :param x: input image tensor
        :param sex: sex tensor
        :param age: age tensor
        :param race: race tensor
        :param result: result tensor
        :param logits: logits tensor
        :return: total_labels: list of entire truth labels from the read in dataset, total_preds: list of
        predictions made from the model, total_logits: array of each inputs model logit outputs from which the
        total_preds were made from (by argmax).
        '''
        self.file_list=gf.create_raw_file_list(train_or_test_path)
        batch_sets=self.batch_set_gen(self.file_list)
        total_labels = []
        total_preds = []
        total_logits = []
        total_files = []
        for file_groups in batch_sets:
            image_batch = []
            label_batch = []
            age_batch = []
            sex_batch = []
            race_batch =[]
            count = 0
            for f in file_groups:
                image, worms, sexes, ages, races = self.getImageFromRaw(f)
                image = (np.expand_dims(image, axis = 0))
                if count == 0:
                    image_batch = image
                    count+=1
                else:
                    image_batch = np.concatenate((image_batch, image), axis = 0)
                label_batch.append(worms)
                age_batch.append(ages)
                sex_batch.append(sexes)
                race_batch.append(races)
                total_files.append(f)


            label_batch = np.asarray(label_batch)
            age_batch = np.asarray(age_batch)
            sex_batch = np.asarray(sex_batch)
            race_batch = np.asarray(race_batch)
            if self.class_type =='binary':
                label_batch[label_batch<=1]=0
            elif self.class_type=='three_class':
                label_batch[label_batch <= 1] = 0
                label_batch[(label_batch > 1) & (label_batch <4)] = 1
                label_batch[label_batch==4] = 2
            elif self.class_type=='four_class':
                label_batch[label_batch <= 1] = 0
                label_batch[label_batch == 2] = 1
                label_batch[label_batch == 3] = 2
                label_batch[label_batch==4] = 3
            elif self.class_type!='':
                sys.exit('%s is not a valid class type. Must be `binary`, `three_class`, or empty.' % self.class_type)
            label_pred, label_logits = sess.run([result, logits], {x:image_batch, sex:sex_batch, age:age_batch, race:race_batch})

            total_labels = np.concatenate((total_labels, label_batch))
            total_preds = np.concatenate((total_preds, label_pred))
            if len(total_logits)==0:
                total_logits=label_logits
            else:
                total_logits=np.concatenate((total_logits, label_logits), axis = 0)
        total_labels = total_labels[0:len(self.file_list)]
        total_preds = total_preds[0:len(self.file_list)]
        total_logits = total_logits[0:len(self.file_list)]
        return total_labels, total_preds, total_logits

    def save_results(self, subset, itt, total_preds, total_labels, total_logits,
                     model, version, new_log, file_ver):
        '''
        
        :param subset: 'train', 'test', or 'valid'
        :param itt: iteration number that the model is at
        :param total_preds: predictions at that iteration
        :param total_labels: truth values for the model
        :param total_logits: logits from current iteration model
        :param model: model name
        :param version: version of model
        :param new_log: where the models terminal output is being saved
        :param file_ver: dictionary number
        :return: dictionary containing information on accuracy, confusion matrix, and logits for a given
        iteration # of the model for training, testing, and validation. As well as the files in the data subset and their
        corresponding truth values.
        '''
        self.dict_save = join('/data/bigbone4/DeepLearning_temp/Doc/', model, str(
            version), 'metric_save' + str(file_ver) + '.pickle')
        self.new_log = new_log
        data_types = ['train', 'valid', 'test']
        if os.path.exists(self.dict_save):
            with open(self.dict_save, 'rb') as f:
                acc_dict = pickle.load(f)
        else:
            acc_dict = {}
            acc_dict['log_file'] = self.new_log
            for s in data_types:
                acc_dict[s] = {}
                acc_dict[s]['results'] = {}

        confmat, train_acc,_ = gf.results_gen(total_labels, total_preds)
        acc_dict[subset]['files'] = self.file_list
        acc_dict[subset]['truth_labels'] = total_labels

        acc_dict[subset]['results'][str(itt)] = {}
        acc_dict[subset]['results'][str(itt)]['logits'] = total_logits
        acc_dict[subset]['results'][str(itt)]['confmat'] = confmat
        acc_dict[subset]['results'][str(itt)]['train_acc'] = train_acc
        with open(self.dict_save, 'wb') as f:
            pickle.dump(acc_dict, f, pickle.HIGHEST_PROTOCOL)


