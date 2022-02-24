import os
import time as the_time_lib
def run_exp(dataset='cifar100', phase=5, tfs=False, weight_pod_loss=1.0, weight_icarl_loss=0.0, weight_lucir_loss=0.0, gpu=0):

    machine = 'volta'

    if tfs:
        if dataset=='cifar100':
            if phase==5:
                the_options = 'options/config_cifar100_5phase_tfs.yaml'
            elif phase==10:
                the_options = 'options/config_cifar100_10phase_tfs.yaml'
            elif phase==25:
                the_options = 'options/config_cifar100_25phase_tfs.yaml' 
            else:
                raise ValueError('Please set correct number of phases.')

        elif dataset=='imagenet_sub':
            if phase==5:
                the_options = 'options/config_imagenet_subset_5phase_tfs.yaml'
            elif phase==10:
                the_options = 'options/config_imagenet_subset_10phase_tfs.yaml'
            elif phase==25:
                the_options = 'options/config_imagenet_subset_25phase_tfs.yaml' 
            else:
                raise ValueError('Please set correct number of phases.')    
        else:
            raise ValueError('Please set correct dataset.') 
    else:
        if dataset=='cifar100':
            if phase==5:
                the_options = 'options/config_cifar100_5phase.yaml'
            elif phase==10:
                the_options = 'options/config_cifar100_10phase.yaml'
            elif phase==25:
                the_options = 'options/config_cifar100_25phase.yaml' 
            else:
                raise ValueError('Please set correct number of phases.')       
        elif dataset=='imagenet_sub':
            if phase==5:
                the_options = 'options/config_imagenet_subset_5phase.yaml'
            elif phase==10:
                the_options = 'options/config_imagenet_subset_10phase.yaml'
            elif phase==25:
                the_options = 'options/config_imagenet_subset_25phase.yaml' 
            else:
                raise ValueError('Please set correct number of phases.')    
        else:
            raise ValueError('Please set correct dataset.')

    the_command = 'python3 -minclearn'
    the_command += ' --options ' + the_options
    the_command += ' --disable-rmm'
    the_command += ' --disable-search'
    the_command += ' --device ' + str(gpu)
    the_command += ' --save_model'
    the_command += ' --weight_pod_loss ' + str(weight_pod_loss)
    the_command += ' --weight_icarl_loss ' + str(weight_icarl_loss)
    the_command += ' --weight_lucir_loss ' + str(weight_lucir_loss)
    the_command += ' --label imagenet_sub_' + str(phase) + 'phase'

    if machine == 'volta':
        the_time = str(the_time_lib.time())
        the_command += ' 2>&1 | tee ' + 'log_' + dataset + '_' + str(phase) + 'phase' + the_time
        os.system(the_command)
    else:
       raise ValueError('Please set correct workstation.')

run_exp(dataset='imagenet_sub', phase=5, gpu=0)
run_exp(dataset='imagenet_sub', phase=10, gpu=0)
run_exp(dataset='imagenet_sub', phase=25, gpu=0)
