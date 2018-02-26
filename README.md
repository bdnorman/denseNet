# **Neural Network Set Up for Automatic KL predictions**
#### Berk Norman

## **Usage**
The main function to actually run the neural network with given parameters is `net_runner.py`. The data to be used for these models muse be MANUALLY SET in `lines 298 - 301` as the *training, testing, validation*, and *training_aug* folders . The batch size and number of epochs to train for are hardcoded in `lines 304 - 305` as 30 and 200, respectively.
Example usage of this function is as follows: <br>
```python
$ python net_runner.py --gpus_vis 1 --gpu_usage 1.0 --modelname KL_w_UNet
    --modeltype dense_net0 --image_size 500 500 1 
    --model_weights 0.8 1.1 1.0 0.8 1.0 --dem_include True
```

#### Data Input Note
Inputted data is of float32 .raw format in the order of kl score, sex, age, race, flattened xray. **IMPORTANT NOTE**: the flattened volume for this were flattened in row-major (C-style) order. If volumes are flattened in MATLAB, there are flattened in column-major (Fortran-style/F-style) order. If this is the case, in `queue_input.py` will need to permute the volume, the command for which has been commented out in `line 40`. Additionally, the *np.reshape* order argument needs to be changed to *'F'* in the `acc_class.py` function `getImageFromRaw`.

### Inputs
**--gpus_vis** *(Not required)*<br>
GPU visible to model if running on a multi-GPU machine. If a single GPU machine, this argument is not required as the default is GPU # 0.<br><br>
**--gpu_usage** *(Not required)*<br>
Float value to indicate what percentage of the GPU should be used during training. This  is useful is the model is small and you want to train multiple model on the same GPU. Default is 1.0, which will use the entire GPU.<br><br>
**--modelname** *(Required)*<br>
General name of project that all output types and folders should be saved under.<br><br>
**--model_type** *(Required)*<br>
Name of the model architecture to use. Current options are *dense_net0*, *alex_net_mini_1*, and *alex_net_mini_2*. Additional architectures can be added to the `architecture.py` file and then the `architecture_helpy.py` file.<br><br>
**--model_params** *(Not required)*<br>
File path to previous models pickle file that Contains input parameters used to train that model. Default is not to load in previous parameters.<br>

**--image_size** *(Required)*<br>
Height, width, and depth integers inputted .raw file images (depth is 1 for grayscale images and 3 for RGB).


**--model_weights** *(Required)*<br>
Floats defining weights to use in loss function. This is useful for class imbalances. If no weighting is needed, weights should all be 1.0 1.0 ... <br>

**--class_type** *(Not required)*<br>
Classification type to make. For full 0-4 class prediction, this argument is not required. For binary class of OA vs. non OA (classes 0-1 vs. 2-4), argument should be `binary`. For three class classification of no OA (score 0-1), mild OA (2-3), and severe OA (4), argument should be `three_class`. Defaults is `''` for original classification.<br>

**--loss** *(Not required)*<br>
Default is using `cross_entropy` loss, which I found has worked best for this problem. Other options are `mse` for using mean-squared error loss and `mse_ce_comb` for an average of the MSE and CE loss, as suggest in https://arxiv.org/abs/1609.02469.

**--restore_model** *(Not required)*<br>
Full directory path to .ckpt model of previousously trained model on the same architecture. Default is to load no model.<br>

**--batch_norm** *(Not required)*<br>
Option to include batch normalization after convolutions. Default is False.<br>

**--dem_include** *(Not required)*<br>
Option to include vector of demographics (age, sex, and race) into the neural network. Default is False.



### Outputs
*doc_folder* = `~/densenet/Doc/`. This is hard coded in `line 106`.<br>
*projects* = args.modelname <br>
*version* is hardcoded on `line 53` and is a unique name string created by combining values of the model inputs.<br>

**models**<br>
Every 20,000 iterations, a model checkpoint is save to *doc_folder/version/models*.  

**log output text**<br>
When running `net_runner.py`, the console will output a combination of loss, confusion matrices, and training, validation, and testing accuracies. This whole output is save as a text file under *doc_folder/version/logs*. If a train_log.txt file already exists for the given specified model and version, the console will alert you that this model type has already been run, do you wish to continue. If user enters [y], the new log files will be saved as *train_log[# of existing files in the directory + 1].txt*. If [n], the program will exit. It [d], the most recent train_log[#].txt file will be deleted and the new one will be saved as train_log[#].txt

**output stats**<br>
As described in detail in the `acc_class.py` `save_results` function, this is a pickle file containing information about the models performances on training, testing, and validation every 20,000 iterations such as that iterations logits, accuracy, and confusion matrix as well as the original file paths and truth values. Saved under *doc_folder/version/metric_save1.pickle*.

**model parameters**
pickle files containing the specified model parameters (model, WEIGHTS, class_type, loss, restore_model, use_batchn, use_dems). Saved as *doc_folder/version/logs/param_dict.p*
