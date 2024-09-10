"""
Author: Mengdie Huang (Maggie)
Address: Xidian University, Purdue University 
Date: 2024-06-13
@copyright
"""

import argparse
import yaml
import datetime
from utils.runid import GetRunID     
import os


import random
import numpy as np
import torch




def parse_arguments():
    parser = argparse.ArgumentParser(description='parse command')
    subparsers = parser.add_subparsers(description='parse subcommand',dest='subcommand')
    
    # load: parse from configure file .yml 
    parser_load = subparsers.add_parser('load',help = 'run command from a ./experiments/yaml file')
    parser_load.add_argument('--config',type=str,default='experiments/experiment-command.yaml')
    
    # run:  parse from keyboard command line
    parser_run = subparsers.add_parser('run',help = 'run command from the command line')
    for parser_object in [parser_load, parser_run]:       
        
        parser_object.add_argument('--taskmode',type=str,default=None, 
            choices=[
                'pretrain_source', 
                'finetune_surrogate', 
                'surrogate_genae',
                'transfer_attack',
                'adapt_target',
                'advtrain',
                'transfer_attack_target'
                ], 
            help="""
                pretrain_source:       prepare the source (foundation) model, load pre-trained foundation model on ImageNet.
                finetune_surrogate:    prepare the surrogate model, fine-tune the pre-trained surrogate model on surrogate dataset.
                transfer_attack:       generate transferable adversarial examples based on surrogate source model
                adapt_target:        produce target (adaptation) model from the source model via transfer learning
                advtrain:              adversarially train the model via pgd-train 
            """)
                  
        parser_object.add_argument('--finetune_mode',type=str,default=None, choices=['all', 'lastlayer'])

        parser_object.add_argument('--source_model',type=str,default=None, 
            choices=[
                'vitb16','vitb32','vitl16','vitl32',
                'convnexttiny','convnextsmall','convnextbase','convnextlarge',
                'convnextv2atto','convnextv2femto','convnextv2pico','convnextv2nano','convnextv2tiny','convnextv2base','convnextv2large','convnextv2huge',
                'clip',
                'blip2'
                ])
        parser_object.add_argument('--surrogate_model',type=str,default=None, 
            choices=[
                'resnet18','resnet34','resnet50','vgg16', 'inceptionv3','densenet121','swinb','swinbv2'
                ])
        parser_object.add_argument('--target_model',type=str,default=None, 
            choices=[
                'mobilenetv2', # 2018
                'mobilenetv3', # 2019
                'efficientnet',
                'vitb16','vitl16','convnextbase','convnextlarge','convnextv2base','convnextv2large'
                ])
        
        parser_object.add_argument('--adapt_target_method',type=str,default=None, 
            choices=[
                'stdtrain',  
                'finetune',  
                'taskdistill',
                'card'
                ])
        parser_object.add_argument('--target',default=False,type = bool)
        parser_object.add_argument('--distill_temperature', type=float, default=3.0)  
        parser_object.add_argument('--distill_lambda', type=float, default=0.7)  
        parser_object.add_argument('--sp_attack',type = str,choices=['pgd-l2','pgd-linf'])
        parser_object.add_argument('--source_model_path',type=str,default=None, help='Load path of source model')
        parser_object.add_argument('--target_model_path',type=str,default=None, help='Load path of target model')
        parser_object.add_argument('--surrogate_model_path',type=str,default=None, help='Load path of surrogate model')
        parser_object.add_argument('--all',action='store_true',help='are you filtering source dataset?')
        parser_object.add_argument('--warmup_lr', action='store_true')
        parser_object.add_argument('--warmup_lr_epoch', type = int,default=20)
        parser_object.add_argument('--source_dataset',type=str,default='imagenet', choices=['imagenet'], help='source-domain dataset')
        parser_object.add_argument('--surrogate_dataset',type=str,default=None, 
                                   choices=['imagenette', 'tiny_imagenet', 'imagewoof','imagenet-10-in','imagenet-10-iw','imagenet-200'], 
                                   help='surrogate dataset')
        parser_object.add_argument('--image',type=bool,default=True)
        parser_object.add_argument('--target_dataset',type=str,default=None, choices=['cifar10', 'cifar100'], help='target-domain dataset')
        parser_object.add_argument('--optim', type=str, default='sgd',
                        choices=['sgd', 'adam', 'adamw','entropySGD' , 'APM'],
                        help='select which optimizer to use')
        parser_object.add_argument('--reg', action='store_true')
        parser_object.add_argument('--reg-type', type=str, default='ig')
        parser_object.add_argument('--ig-beta', type=float, default=100.)
        parser_object.add_argument('--sam', action='store_true')
        parser_object.add_argument('--transattacksurrogate',action='store_true')
        parser_object.add_argument('--rho', type=float, default=0.05, help='sam parameters')
        parser_object.add_argument('--data_path',type=str,default=None, help='Load Path of dataset')
        parser_object.add_argument('--surrogate_advdataset_x_path',type=str,default=None, help='Load Path of surrogate_advdataset_x_path')
        parser_object.add_argument('--surrogate_advdataset_y_path',type=str,default=None, help='Load Path of surrogate_advdataset_y_path')
        parser_object.add_argument('--lr_schedule', type=str, default='piecewise', choices=['cosine', 'piecewise', 'constant'], help='learning schedule')
        parser_object.add_argument('--robust',action='store_true')
        parser_object.add_argument('--usingsp', action='store_true')
        parser_object.add_argument('--n_classes',type=int,default=None,help='Number of classes')
        parser_object.add_argument('--n_features',type=int, default=None,help='Number of input features')
        parser_object.add_argument('--n_epochs',type=int,default=None,help='Epochs number')
        parser_object.add_argument('--batch_size',type=int,default=None,help='Batch size')
        parser_object.add_argument('--lr',type=float,default=0.1,help='Intial learning rate')
        parser_object.add_argument('--weight-decay', type=float, default=1e-4,
                        help='set the weight decay rate')
        parser_object.add_argument('--momentum', type=float, default=0.9,
                        help='set the momentum for SGD')
        parser_object.add_argument('--n_cpus',type=int,default=None,help='Number of CPUs to use')
        parser_object.add_argument('--n_gpus',type=int,default=None,help='Number of GPUS to use')
        parser_object.add_argument('--seed', type=int, default=42)  
        # parser_object.add_argument('--gamma', type=float, default=1, help='weight for classification')
        # parser_object.add_argument('--cof',type=float,default=1)
        # parser_object.add_argument('-b', '--beta', type=float, default=.5, help='weight balance for other losses')
        parser_object.add_argument('--task_name',type = str)
        # NCE distillation
        parser_object.add_argument('--norm', default='inf')
        parser_object.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
        parser_object.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
        parser_object.add_argument('--model_folder',default='result/Contrastive')
        parser_object.add_argument('--nce_k', default=2048, type=int, help='number of negative samples for NCE')
        parser_object.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
        parser_object.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')
        parser_object.add_argument('--is_sample', action='store_true')
        parser_object.add_argument('--save_path',type=str,default='result',help='Output path for saving results')
        parser_object.add_argument('--input_dim',default=(3,224,224))
        parser_object.add_argument('--n_eval_step',type = int, default=10)
        parser_object.add_argument('--attack_type', help=' transfer attack method', type=str, default=None, 
                                   choices=['fgsm', 'pgd-linf', 'pgd-l2', 'pgd-l1', 'mifgsm-linf', 'mifgsm-l2', 'fab-linf', 'fab-l2', 'fab-l1', 'cw-l2', 'autoattack', 'deepfool','universal','zoo','boundary','hopskipjump','simba','square'])
        parser_object.add_argument('--pgd-random-start', action='store_true',
                        help='if select, randomly choose starting points each time performing pgd')
        parser_object.add_argument('--ae_eps', help='epsion (pixel/255) of the adversarial exmaple', type=int, default=16)
        parser_object.add_argument('--ae_step', help='step size (pixel/255) of the adversarial exmaple', type=int, default=4)        
        parser_object.add_argument('--ae_ite', help='iteration number of the adversarial exmaple', type=int, default=20)
        parser_object.add_argument('--gamma', default=.6,type = float)
        parser_object.add_argument('--beta',default=.4,type = float)        
        parser_object.add_argument('--cof', default=0)
    return parser.parse_args()

def check_arguments(args):

    if  args.save_path == None:
        raise Exception('args.save_path=%s, please input the save_path' % args.save_path)   
    if  args.seed == None:
        raise Exception('args.seed=%s, please input the seed' % args.seed)   
    
    if  args.taskmode == None:
        raise Exception('args.taskmode=%s, please input the taskmode' % args.taskmode)   
    else:
        if args.taskmode == 'pretrain_source':
            if  args.source_model == None:
                raise Exception('args.source_model=%s, please input the source_model' % args.source_model)   
            if  args.source_dataset == None:
                raise Exception('args.source_dataset=%s, please input the source_dataset' % args.source_dataset)   
        
        elif args.taskmode == 'finetune_surrogate':
            if  args.surrogate_model == None:
                raise Exception('args.surrogate_model=%s, please input the surrogate_model' % args.surrogate_model)   
            if  args.surrogate_dataset == None:        
                raise Exception('args.surrogate_dataset=%s, please input the surrogate_dataset' % args.surrogate_dataset)
            if args.finetune_mode == None:
                raise Exception('args.finetune_mode=%s, please input the finetune_mode' % args.finetune_mode)
        
        elif args.taskmode == 'surrogate_genae':
            if  args.surrogate_model == None:
                raise Exception('args.surrogate_model=%s, please input the surrogate_model' % args.surrogate_model)   
            if  args.surrogate_dataset == None:        
                raise Exception('args.surrogate_dataset=%s, please input the surrogate_dataset' % args.surrogate_dataset)
            if  args.attack_type == None:
                raise Exception('args.attack_type=%s, please input the attack_type' % args.attack_type)
            if  args.ae_eps == None:
                raise Exception('args.ae_eps=%s, please input the ae_eps' % args.ae_eps)
            if  args.ae_step == None:
                raise Exception('args.ae_step=%s, please input the ae_step' % args.ae_step)
            if  args.ae_ite == None:
                raise Exception('args.ae_ite=%s, please input the ae_ite' % args.ae_ite)
            
        elif args.taskmode == 'transfer_attack':
            if  args.source_model == None:
                raise Exception('args.source_model=%s, please input the source_model' % args.source_model)   
            # if  args.source_dataset == None:
            #     raise Exception('args.source_dataset=%s, please input the source_dataset' % args.source_dataset)         
            if  args.surrogate_model == None:
                raise Exception('args.surrogate_model=%s, please input the surrogate_model' % args.surrogate_model)   
            if  args.surrogate_dataset == None:
                raise Exception('args.surrogate_dataset=%s, please input the surrogate_dataset' % args.surrogate_dataset)
            if  args.attack_type == None:
                raise Exception('args.attack_type=%s, please input the attack_type' % args.attack_type)
            if  args.surrogate_advdataset_x_path == None:
                raise Exception('args.surrogate_advdataset_x_path=%s, please input the surrogate_advdataset_x_path' % args.surrogate_advdataset_x_path)
            if  args.surrogate_advdataset_y_path == None:
                raise Exception('args.surrogate_advdataset_y_path=%s, please input the surrogate_advdataset_y_path' % args.surrogate_advdataset_y_path)
            
        elif args.taskmode == 'transfer_attack_target':
            if  args.target_model == None:
                raise Exception('args.source_model=%s, please input the source_model' % args.target_model)   
            # if  args.source_dataset == None:
            #     raise Exception('args.source_dataset=%s, please input the source_dataset' % args.source_dataset)         
            if  args.surrogate_model == None:
                raise Exception('args.surrogate_model=%s, please input the surrogate_model' % args.surrogate_model)   
            if  args.surrogate_dataset == None:
                raise Exception('args.surrogate_dataset=%s, please input the surrogate_dataset' % args.surrogate_dataset)
            if  args.attack_type == None:
                raise Exception('args.attack_type=%s, please input the attack_type' % args.attack_type)
            if  args.surrogate_advdataset_x_path == None:
                raise Exception('args.surrogate_advdataset_x_path=%s, please input the surrogate_advdataset_x_path' % args.surrogate_advdataset_x_path)
            if  args.surrogate_advdataset_y_path == None:
                raise Exception('args.surrogate_advdataset_y_path=%s, please input the surrogate_advdataset_y_path' % args.surrogate_advdataset_y_path)

        elif args.taskmode == 'adapt_target':
            if  args.source_model == None:
                raise Exception('args.source_model=%s, please input the source_model' % args.source_model)   
            if  args.source_model_path == None:
                raise Exception('args.source_model_path=%s, please input the source_model_path' % args.source_model_path)   
            if  args.target_model == None:
                raise Exception('args.target_model=%s, please input the target_model' % args.target_model)   
            if  args.target_dataset == None:
                raise Exception('args.target_dataset=%s, please input the target_dataset' % args.target_dataset)                    
            if  args.source_dataset == None:
                raise Exception('args.source_dataset=%s, please input the source_dataset' % args.source_dataset)    
            if  args.target_dataset == None:
                raise Exception('args.target_dataset=%s, please input the target_dataset' % args.target_dataset)             
        else:
            raise Exception('args.taskmode=%s is illegal, please input the correct taskmode' % args.taskmode)

def set_exp_result_dir(args):

    cur=datetime.datetime.now()
    date = f'{cur.year:04d}{cur.month:02d}{cur.day:02d}'
    
    if args.taskmode == 'pretrain_source':
        args.save_path=f'{args.save_path}/{args.taskmode}/{args.source_model}/{args.source_dataset}/seed{args.seed:02d}/{date}' 
        
    if args.taskmode == 'finetune_surrogate':
        args.save_path=f'{args.save_path}/{args.taskmode}/{args.surrogate_model}/{args.surrogate_dataset}/{args.finetune_mode}/seed{args.seed:02d}/{date}'  
        
    if args.taskmode == 'surrogate_genae':
        if not args.transattacksurrogate:
            args.save_path=f'{args.save_path}/{args.taskmode}/{args.surrogate_model}/{args.surrogate_dataset}/{args.attack_type}/{args.ae_eps}-{args.ae_step}-{args.ae_ite}/seed{args.seed:02d}/{date}'   
        else:    
            args.save_path=f'{args.save_path}/{args.taskmode}/{args.surrogate_model}_sp/{args.surrogate_dataset}/{args.attack_type}/{args.ae_eps}-{args.ae_step}-{args.ae_ite}/seed{args.seed:02d}/{date}'   

    
    if args.taskmode == 'transfer_attack':
        args.save_path=f'{args.save_path}/{args.taskmode}/{args.source_model}/{args.source_dataset}/{args.surrogate_model}/{args.surrogate_dataset}/{args.attack_type}/{args.ae_eps}-{args.ae_step}-{args.ae_ite}/seed{args.seed:02d}/{date}'   
        
    if args.taskmode == 'adapt_target':
        if args.adapt_target_method == 'finetune':
            args.save_path=f'{args.save_path}/{args.taskmode}/{args.adapt_target_method}/{args.finetune_mode}/{args.source_model}/{args.source_dataset}/{args.target_model}/{args.target_dataset}/seed{args.seed:02d}/{date}'             
        else:
            args.save_path=f'{args.save_path}/{args.taskmode}/{args.adapt_target_method}/{args.source_model}/{args.source_dataset}/{args.target_model}/{args.target_dataset}/seed{args.seed:02d}/{date}' 
         
    if args.taskmode == 'transfer_attack_target':
        args.save_path=f'{args.save_path}/{args.taskmode}/{args.source_model}/{args.source_dataset}/{args.surrogate_model}/{args.surrogate_dataset}/{args.attack_type}/{args.ae_eps}-{args.ae_step}-{args.ae_ite}/seed{args.seed:02d}/{date}' 
        
        
    cur_run_id = GetRunID(args.save_path)
    args.save_path = os.path.join(args.save_path, f'{cur_run_id:03d}')    

    return args.save_path


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def parse_argu(): 

    args = parse_arguments()
    args_dictionary = vars(args)
    check_arguments(args)
    args_dictionary_yaml = yaml.dump(args_dictionary)    
    set_seed(args.seed)
    args.save_path = set_exp_result_dir(args)
    os.makedirs(args.save_path, exist_ok=True)

    exp_yaml=open(f'{args.save_path}/exp-command.yaml', "w")    
    exp_yaml.write(args_dictionary_yaml)

    return args


