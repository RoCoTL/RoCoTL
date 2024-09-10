
import numpy as np
import torch
import torchvision 
import torchvision.transforms as transforms
import torch.nn as nn
import torch.backends.cudnn as cudnn
import pandas
from datetime import datetime
from torchvision.models import wide_resnet50_2
import torch.optim as optim
import pickle
import time
import os
from art.attacks.evasion import ProjectedGradientDescent
import sys
from utils.eval import evaluate_model
from utils.draw import drawplt_line

from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F 
import types
from utils.ealystop import EarlyStopping

sys.path.append('./newcard')
from newcard.algorithm.args import *
from newcard.algorithm.tools import AttackPGD,evaluate,generate_atk,accuracy,fpandfn1,precision_score1, recall_score1, f1_score1
from newcard.cardv2.helper.loopsv2 import validate,train_distill,DistillKL
from newcard.cardv2.crd.criterionv2 import CRDLoss


def lr_decay(epoch, total_epoch,args):
        if args.lr_schedule == 'piecewise':
            if total_epoch == 200:
                epoch_point = [100, 150]
            elif total_epoch == 110: 
                epoch_point = [100, 105] # Early stop for Madry adversarial training
            elif total_epoch ==50:
                epoch_point = [25, 40]
            else:
                epoch_point = [int(total_epoch/2),int(total_epoch-5)]
            if epoch < epoch_point[0]:
                if args.warmup_lr and epoch < args.warmup_lr_epoch:
                    return 0.001 + epoch / args.warmup_lr_epoch * (args.lr-0.001)
                return args.lr
            if epoch < epoch_point[1]:
                return args.lr / 10
            else:
                return args.lr / 100
        elif args.lr_schedule == 'cosine':
            if args.warmup_lr:
                if epoch < args.warmup_lr_epoch:
                    return 0.001 + epoch / args.warmup_lr_epoch * (args.lr-0.001)
                else:
                    return np.max([args.lr * 0.5 * (1 + np.cos((epoch-args.warmup_lr_epoch) / (total_epoch-args.warmup_lr_epoch) * np.pi)), 1e-4])
            return np.max([args.lr * 0.5 * (1 + np.cos(epoch / total_epoch * np.pi)), 1e-4])
        elif args.lr_schedule == 'constant':
            return args.lr
        else:
            raise NotImplementedError  
        
def forward(self, x: torch.Tensor):
    # Reshape and permute the input tensor
    x = self._process_input(x)
    n = x.shape[0]

    # Expand the class token to the full batch
    batch_class_token = self.class_token.expand(n, -1, -1)
    x = torch.cat([batch_class_token, x], dim=1)

    x = self.encoder(x)

    # Classifier "token" as used by standard language architectures
    x = x[:, 0]
    feature = x
    x = self.heads(x)

    return x , feature

# def card(model_s,model_t,train_loader,test_loader,criterion,numofclass,device,args):
def card(model_s,model_t,train_loader,test_loader,criterion,numofclass,device,args):

    #---------maggie add-----------
    print("device:",device) #device: cuda
    model_s = model_s.to(device)  
    model_t = model_t.to(device) 
    #---------maggie add-----------
    early_stopping = EarlyStopping(save_path=args.save_path, patience=50) 

    best_acc = 0
    opt = torch.optim.SGD(model_s.parameters(), args.lr, 
                            weight_decay=args.weight_decay,
                            momentum=args.momentum)
    minv = 0
    maxv = 0
    # Get the shape of a single sample
    n_data = int(len(train_loader.dataset) /0.8)
    # n_data: 40000 because cifar-10 trainingset is 4k, validation is 1k, total 5k
    
    # Combine to infer overall shape
    overall_shape = 0
    if not args.is_sample: 
        for data, label in train_loader:
            x = torch.max(data).item()
            y = torch.min(data).item()
            overall_shape = data.shape[1:]
            maxv = x if max == 0 else max(maxv, x)
            minv = y if min == 0 else min(minv, y)
    else:
        for data, label,index,contraindex in train_loader:
            x = torch.max(data).item()
            y = torch.min(data).item()
            overall_shape = data.shape[1:]
            maxv = x if max == 0 else max(maxv, x)
            minv = y if min == 0 else min(minv, y)
    
    # print(minv,maxv)
    # print("minv:", minv)
    # print("maxv:", maxv)
    """ 
    minv: -2.1179039478302
    maxv: 2.640000104904175
    """
    
    
    model_t.forward = types.MethodType(forward, model_t)
    classifier = generate_atk(model_s,criterion=criterion,optimizer=opt,min=minv,max=maxv,shape= overall_shape,number=numofclass)
    attack = ProjectedGradientDescent(estimator=classifier,norm=args.norm, eps=args.ae_eps/255,eps_step=args.ae_step/255,targeted = args.target,max_iter= args.ae_ite)
    
    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)
    
    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = nn.KLDivLoss()
    batch = next(iter(train_loader))
    s = batch[0].shape[1:]
    
    # print(s)
    # print("s:", s)
    """ 
    s: torch.Size([3, 224, 224])
    """
    
    
    data = torch.rand(1, *s)               

    
    
    
    
    # print("data.shape:", data.shape)
    """ 
    data.shape: torch.Size([1, 3, 224, 224])
    """
    

    
    pholder = model_s(data.to(device))
    
     
    # print("pholder:",pholder)
    # print("pholder.shape:", pholder.shape)
    # print("pholder.is_cuda:",pholder.is_cuda)
    # print("pholder.get_device():",pholder.get_device())
    """ 
    pholder: tensor([[ 1.0323, -2.7018, -3.3212, -0.4711, -2.8097,  0.0457,  0.2004, -2.4782,
            0.8326,  2.1257]], device='cuda:0', grad_fn=<AddmmBackward0>)
    pholder.shape: torch.Size([1, 10])
    pholder.is_cuda: True
    pholder.get_device(): 0
    """
      
    
    args.s_dim = model_s.feature.shape[1]
    args.n_data = n_data
    # print("args.s_dim:",args.s_dim)
    # print("args.n_data:",args.n_data)
    """
    args.s_dim: 2048
    args.n_data: 40000
    """


    
    data=data.to(device)
    # print("data.shape:", data.shape)
    # print("data.is_cuda:",data.is_cuda)
    # print("data.get_device():",data.get_device())
    """ 
    data.shape: torch.Size([1, 3, 224, 224])
    data.is_cuda: True
    data.get_device(): 0
    model_t is on cuda: 0
    """
    
    
    _,feature = model_t(data) 
    
    args.t_dim = feature.shape[1]
    # print("args.t_dim:",args.t_dim)
    # args.t_dim: 768
    
        
    criterion_kd = CRDLoss(args,s)          # maggie: error from this contrastive loss
    module_list.append(criterion_kd.embed_s)
    module_list.append(criterion_kd.embed_t)
    trainable_list.append(criterion_kd.embed_s)
    trainable_list.append(criterion_kd.embed_t)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)     # other knowledge distillation loss
    print(len(trainable_list))
    # optimizer
    optimizer = optim.SGD(trainable_list.parameters(),
                        lr=args.lr,
                        momentum=args.momentum,
                        weight_decay=args.weight_decay)

    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True

    # routine
    std = []
    adv = []
    train_losses = []
    val_losses = []
    top1_accs = []
    top5_accs = []
    epoch_times = []

    dataframetime = pandas.DataFrame({'seed':[],'epochs':[],'time':[]})
    time0 = time.time()
    for epoch in range(1, args.n_epochs + 1):
        lr_current = lr_decay(epoch, args.n_epochs,args)
        optimizer.param_groups[0].update(lr=lr_current)
        
        print(f"==> {epoch}/{args.n_epochs}-epoch training start...")
        # print("erro happened after this line ---maggie")

        time1 = time.time()
        # easy to say CUDA out of memory
        
        
        #======for quick test=========
        train_acc, train_loss,df = train_distill(epoch, train_loader, module_list, criterion_list, optimizer, args,dataframetime, s, numofclass)
        # train_acc = 0
        # train_loss = 100
        # df = 10
        #======for quick test=========
        
        
        
        time2 = time.time()
        
        dataframetime = df
        print('epoch {}, total time {:.2f} trainacc{}'.format(epoch, time2 - time1,train_acc))

        epoch_time = time2 - time1
        epoch_times.append(epoch_time)
        
        #=============maggie add #=============
        # test_acc, adv_test, test_loss = validate(test_loader, model_s, criterion_cls, numofclass, args,attack,numofclass)

        top1_acc, top5_acc, val_loss, evaluate_time = evaluate_model(model_s, test_loader, criterion_cls)
        print(f'[{epoch:04d} epoch] Epoch Time: {epoch_time:.2f} seconds, Train Loss: {train_loss:.3f}, Validation Loss: {val_loss:.3f}, Top1 Accuracy: {top1_acc:.3f}, Top5 Accuracy: {top5_acc:.3f}')
        # print(f"Test Accuracy:{test_acc}, Test Loss:{test_loss}, Adv Accuracy: {adv_test} ")
        #=============maggie add #=============

        # if (epoch+1)% args.n_eval_step ==0:
        #     std.append(test_acc)
        #     adv.append(adv_test)
        #     with open('Historyforplotting/card'+"/standardhistory.pkl","wb") as file:
        #         pickle.dump(std,file)
        #     with open('Historyforplotting/card' +"/advhistory.pkl","wb") as file:
        #         pickle.dump(adv,file)

        # #=============maggie change #=============
        # # save the best model
        # test_acc = top1_acc
        # if test_acc > best_acc:

        #     best_acc = test_acc
        #     state = {
        #         'epoch': epoch,
        #         'model': model_s.state_dict(),
        #         'best_acc': best_acc,
        #     }
        #     save_file = os.path.join(args.model_folder, '{}_gamma{}_beta{}_cof{}best_on{}.pth'.format(args.target_model,args.gamma,args.beta,args.cof,args.target_dataset))
        #     print('saving the best model!')
        #     torch.save(state, save_file)
        # #=============maggie change #=============
           

        #=============maggie add #=============
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        top1_accs.append(top1_acc)
        top5_accs.append(top5_acc)
        epoch_times.append(epoch_time)
        
        early_stopping(val_loss, model_s)
        if early_stopping.early_stop == True:
            print("Early stopping")
            model_savepath = f'{args.save_path}/{args.target_model}-{args.target_dataset}-{args.adapt_target_method}-epoch-{epoch:04d}-top1acc-{top1_acc:.3f}-top5acc-{top5_acc:.3f}.hdf5'
            torch.save(model_s, model_savepath)
            break 

                        
        # drawplt_line(x=range(1, args.n_epochs+1), y=train_losses, xlabel='Epoch', ylabel='Training Loss', title=f'card-{args.source_model}-to-{args.target_model}-{args.target_dataset}', savepth=args.save_path)
        # drawplt_line(x=range(1, args.n_epochs+1), y=val_losses, xlabel='Epoch', ylabel='Validation Loss', title=f'card-{args.source_model}-to-{args.target_model}-{args.target_dataset}', savepth=args.save_path)
        # drawplt_line(x=range(1, args.n_epochs+1), y=top1_accs, xlabel='Epoch', ylabel='Validation Top-1 Accuracy', title=f'card-{args.source_model}-to-{args.target_model}-{args.target_dataset}', savepth=args.save_path)
        # drawplt_line(x=range(1, args.n_epochs+1), y=top5_accs, xlabel='Epoch', ylabel='Validation Top-5 Accuracy', title=f'card-{args.source_model}-to-{args.target_model}-{args.target_dataset}', savepth=args.save_path)
        # # drawplt_line(x=range(1, args.n_epochs+1), y=epoch_times, xlabel='Epoch', ylabel='Time Cost (seconds)', title=f'card-{args.source_model}-to-{args.target_model}-{args.target_dataset}', savepth=args.save_path)
                            
        #=============maggie add #=============
        
        # scheduler.step()
        # current_lr = optimizer.param_groups[0]['lr']
        # print(f'Epoch {epoch_index}: current learning rate = {current_lr}')
            
            
            
            
            
    time3 = time.time()
    print(f'total time used: {time3 - time0}')
    print('best accuracy:', best_acc)
    # dataframetime.to_csv(f'result/Contrastive/recordfortime_seed{args.seed}_task_{args.task_name}',index=False)
    # save model
    # state = {
    #     'opt': args,
    #     'model': model_s.state_dict(),
    # }
    # save_file = os.path.join(args.model_folder, '{}_gamma{}_beta{}last_on{}.pth'.format(args.target_model,args.gamma,args.beta,args.target_dataset))
    # torch.save(state, save_file)





    card_time = sum(epoch_times)
    print('Finished card')

    return card_time
