import architecture as models
import sys

def model_selector(model_string, input_tensor, age,sex, race, NUM_OF_CLASSES, BATCH_SIZE, growth_rate, batch_normalization,  prob, include_dems):
    '''

    :param model_string: string indicating which model should be used from the architecture.py module
    :param input_tensor: image tensor of shape [BATCH, HEIGHT, WIDTH, CHANNELS]
    :param age: age tensor of shape [BATCH, 1]
    :param sex: sex tensor of shape [BATCH, 1]
    :param race: race tensor of shape [BATCH, 1]
    :param NUM_OF_CLASSES: number of classifcation groups to make
    :param BATCH_SIZE: batch size being trained with
    :param growth_rate: number of channels to use in dense blocks
    :param batch_norm: boolean on whether or not to use batch normalization after convolutions. Default is false.
    :param prob: dropout rate to keep. Defaulted to 0.5 in net_runner.py and to 1.0 in acc_class.py
    :param include_dems: boolean on whether or not to include inputted demographic vectore. Default is false.
    :return: results and logits from model, where results is just the argmax across that channel dimension of the logits.
    '''
    if model_string=='dense_net0':
        return models.dense_net0(input_tensor, age, sex, race, NUM_OF_CLASSES, BATCH_SIZE, growth_rate, batch_normalization, include_dems)
    elif model_string=='alex_net_mini_1':
        return models.alex_net_mini_1(input_tensor, age, sex, race, NUM_OF_CLASSES, BATCH_SIZE, batch_normalization, prob, include_dems)
    elif model_string=='alex_net_mini_2':
        return models.alex_net_mini_1(input_tensor, age, sex, race, NUM_OF_CLASSES, BATCH_SIZE, batch_normalization, prob, include_dems)
    else:
        sys.exit('Model architecture %s does not exists. Must be: `dense_net0`, `alex_net_mini_1`, or `alex_net_mini_2`.' % model_string)
