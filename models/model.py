import torch
# import torchvision
# from models.vit import ViT
from torchsummary import summary
import torchvision.models as torchmodels
from models import convnextv2 
# from models import clipmodel
import models.oriclipmodel.model as clipmodel
import models.oriblipmodel.blip as blip2model
import time
import os
import torch.nn as nn
from torch.nn.utils import parameters_to_vector
from utils.eval import evaluate_model
from tqdm import tqdm
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from utils.ealystop import EarlyStopping
import torchattacks
from utils.tensorboarddraw import line_chart
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset.datapro import get_surrogate_dataloader
import torch.utils.data as Data
from utils.minmaxvalue import get_dataset_min_max
from art.attacks.evasion import AutoAttack
from art.estimators.classification import PyTorchClassifier
import TransferAttackSurrogates.TransferAttack.CIFAR_Train.utils as utils
from models.mobilenetv2 import mobilenet_v2
from models.efficientnetb5 import efficientnet_b5
from models.mobilev3small import mobilenet_v3_small
# from transformers import AutoImageProcessor, ConvNextV2ForImageClassification
# from transformers import AutoImageProcessor, ConvNextV2Model

def get_input_gradient_loss(loss, x):
    grads = torch.autograd.grad(loss, inputs=x, create_graph=True)
    grads = parameters_to_vector(grads)
    return torch.norm(grads)
class AverageMeter():
    def __init__(self):
        self.cnt = 0
        self.sum = 0
        self.mean = 0

    def update(self, val, cnt=1):
        self.cnt += cnt
        self.sum += val * cnt
        self.mean = self.sum / self.cnt

    def average(self):
        return self.mean
    
    def total(self):
        return self.sum
class TARFAModel:
    def __init__(self, args) -> None:                 
        self.args = args
        if not args.transattacksurrogate:
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction='sum')
        if self.args.taskmode == 'pretrain_source':
            if self.args.source_model_path == None:
                self.model = self.__GetModel__()
            else:
                self.model = self.__LoadModel__()

        if self.args.taskmode in ['finetune_surrogate','surrogate_genae']:
            if self.args.surrogate_model_path == None:
                if not args.transattacksurrogate:
                    self.model = self.__GetModel__()
                else:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
                    torch.cuda.empty_cache()
                    best_acc = 0
                    train_loader, val_loader, test_loader = get_surrogate_dataloader(args=args, clean=True)
                    
                    model = self.__GetModel__()
                    model = model.to(device)
                    optimizer = utils.get_optim(args.optim, model, args,
            lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum,sam=args.sam, rho=args.rho)
                    if args.robust:
                        best_robust_acc = 0
                        if args.sp_attack == 'pgd-l2':
                            attack = torchattacks.PGDL2
                            attacker = attack(model=model, eps=args.ae_eps/255, alpha=args.ae_step/255, steps=args.ae_ite,
                                            random_start=args.pgd_random_start)
                        elif args.sp_attack == 'pgd-linf':
                            attack = torchattacks.PGD
                            attacker = attack(model=model, eps=args.ae_eps / 255, alpha=args.ae_step / 255, steps=args.ae_ite,
                                            random_start=args.pgd_random_start)
                        else:
                            raise NotImplementedError

                    
                    self.criterion = self.criterion.to(device)
                    for epoch in tqdm(range(args.n_epochs)):
                        acc_meter = AverageMeter()
                        loss_meter = AverageMeter()

                        pbar = tqdm(train_loader, total=len(train_loader))

                        for x, y in pbar:    
                            x, y = x.to(device), y.to(device)
                            
                            optimizer.zero_grad()
                            if args.robust:
                                model.eval()
                                x = attacker(x, y)

                            model.train()

                            if not args.reg:
                                def closure():
                                    logits = model(x)
                                    loss = self.criterion(logits, y).mean()
                                    loss.backward()
                                    return loss, logits

                            else:
                                x.requires_grad = True
                                def closure():
                                    logits = model(x)
                                    loss = self.criterion(logits, y)
                                    if args.reg_type == 'ig':
                                        assert args.reg_type == 'ig'
                                        loss += args.ig_beta * get_input_gradient_loss(loss, x)
                                        loss /= x.shape[0]
                                    loss.backward()
                                    nn.utils.clip_grad_norm_(model.parameters(),10)
                                    return loss, logits


                            loss, logits = closure()

                            if not args.sam:
                                optimizer.step()
                            else:
                                optimizer.step(closure=closure)

                            _, predicted = torch.max(logits.data, 1)

                            acc = (predicted == y).sum().item() / y.size(0)
                            acc_meter.update(acc)
                            loss_meter.update(loss.item())
                            pbar.set_description("Train acc %.2f Loss: %.2f" % (acc_meter.mean * 100, loss_meter.mean))
                        

                        test_acc, test_loss = utils.evaluate(model, torch.nn.CrossEntropyLoss(), test_loader, device)
                        print(f"Current Epoch clean test accuracy: {test_acc}, Loss : {test_loss}")
                        if args.robust:
                            robust_acc, robust_loss = utils.robust_evaluate(model, self.criterion, test_loader, attacker, device)
                            print(f"Current Epoch robust accuracy: {robust_acc}, Loss : {robust_loss}")

                            if robust_acc > best_robust_acc:
                                best_robust_acc = robust_acc
                                if not os.path.exists(f"./tarobustmodel/beta{args.ig_beta}_rho{args.rho}_{args.surrogate_model}"):
                                    os.makedirs(f"./tarobustmodel/beta{args.ig_beta}_rho{args.rho}_{args.surrogate_model}")
                                torch.save(model.state_dict(),f"./tarobustmodel/beta{args.ig_beta}_rho{args.rho}_{args.surrogate_model}/bestsurrogaterobusttamodel.pth")
                        if test_acc > best_acc:
                            best_acc = test_acc
                            if not os.path.exists(f"./tamodel/beta{args.ig_beta}_rho{args.rho}_{args.surrogate_model}"):
                                os.makedirs(f"./tamodel/beta{args.ig_beta}_rho{args.rho}_{args.surrogate_model}")
                            torch.save(model.state_dict(),f"./tamodel/beta{args.ig_beta}_rho{args.rho}_{args.surrogate_model}/bestsurrogatetamodel.pth")
                    self.model =model
            else:
                if not args.transattacksurrogate:
                    self.model = self.__LoadModel__()    
                else: 
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
                    torch.cuda.empty_cache()
                    best_acc = 0
                    train_loader, val_loader, test_loader = get_surrogate_dataloader(args=args, clean=True)
                    
                    model = self.__LoadModel__()
                    model = model.to(device)
                    optimizer = utils.get_optim(args.optim, model, args,
            lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum,sam=args.sam, rho=args.rho)
                    if args.robust:
                        best_robust_acc = 0
                        if args.sp_attack == 'pgd-l2':
                            attack = torchattacks.PGDL2
                            attacker = attack(model=model, eps=args.ae_eps/255, alpha=args.ae_step/255, steps=args.ae_ite,
                                            random_start=args.pgd_random_start)
                        elif args.sp_attack == 'pgd-linf':
                            attack = torchattacks.PGD
                            attacker = attack(model=model, eps=args.ae_eps / 255, alpha=args.ae_step / 255, steps=args.ae_ite,
                                            random_start=args.pgd_random_start)
                        else:
                            raise NotImplementedError

                    
                    self.criterion = self.criterion.to(device)
                    for epoch in tqdm(range(args.n_epochs)):
                        acc_meter = AverageMeter()
                        loss_meter = AverageMeter()

                        pbar = tqdm(train_loader, total=len(train_loader))

                        for x, y in pbar:    
                            x, y = x.to(device), y.to(device)
                            
                            optimizer.zero_grad()
                            if args.robust:
                                model.eval()
                                x = attacker(x, y)

                            model.train()

                            if not args.reg:
                                def closure():
                                    logits = model(x)
                                    loss = self.criterion(logits, y).mean()
                                    loss.backward()
                                    return loss, logits

                            else:
                                x.requires_grad = True
                                def closure():
                                    logits = model(x)
                                    loss = self.criterion(logits, y)
                                    if args.reg_type == 'ig':
                                        assert args.reg_type == 'ig'
                                        loss += args.ig_beta * get_input_gradient_loss(loss, x)
                                        loss /= x.shape[0]
                                    loss.backward()
                                    nn.utils.clip_grad_norm_(model.parameters(),10)
                                    return loss, logits


                            loss, logits = closure()

                            if not args.sam:
                                optimizer.step()
                            else:
                                optimizer.step(closure=closure)

                            _, predicted = torch.max(logits.data, 1)

                            acc = (predicted == y).sum().item() / y.size(0)
                            acc_meter.update(acc)
                            loss_meter.update(loss.item())
                            pbar.set_description("Train acc %.2f Loss: %.2f" % (acc_meter.mean * 100, loss_meter.mean))
                        

                        
                        metrics = evaluate_model(model, test_loader, torch.nn.CrossEntropyLoss())
                        test_acc = metrics[0]
                        print(f'Top1 Accuracy on {args.surrogate_dataset} clean test set: {metrics[0] * 100:.3f}%')
                        print(f'Top5 Accuracy on {args.surrogate_dataset} clean test set: {metrics[1] * 100:.3f}%')
                        print(f'Average Loss on {args.surrogate_dataset} clean test set: {metrics[2]:.3f}')
                        
                        if args.robust:
                             
                            metrics = evaluate_model(model, test_loader, torch.nn.CrossEntropyLoss(reduction='sum'),attacker)
                            robust_acc= metrics[0]
                            print(f'Top1 Robust Accuracy on {args.surrogate_dataset} clean test set: {metrics[0] * 100:.3f}%')
                            print(f'Top5 Robust Accuracy on {args.surrogate_dataset} clean test set: {metrics[1] * 100:.3f}%')
                            print(f'Average Loss on {args.surrogate_dataset} clean test set: {metrics[2]:.3f}')

                            if robust_acc > best_robust_acc:
                                best_robust_acc = robust_acc
                                if not os.path.exists(f"./tarobustmodel/beta{args.ig_beta}_rho{args.rho}_{args.surrogate_model}_{args.surrogate_dataset}"):
                                    os.makedirs(f"./tarobustmodel/beta{args.ig_beta}_rho{args.rho}_{args.surrogate_model}_{args.surrogate_dataset}")
                                torch.save(model.state_dict(),f"./tarobustmodel/beta{args.ig_beta}_rho{args.rho}_{args.surrogate_model}_{args.surrogate_dataset}/bestsurrogaterobusttamodel.pth")
                        if test_acc > best_acc:
                            best_acc = test_acc
                            if not os.path.exists(f"./tamodel/beta{args.ig_beta}_rho{args.rho}_{args.surrogate_model}_{args.surrogate_dataset}"):
                                os.makedirs(f"./tamodel/beta{args.ig_beta}_rho{args.rho}_{args.surrogate_model}_{args.surrogate_dataset}")
                            torch.save(model.state_dict(),f"./tamodel/beta{args.ig_beta}_rho{args.rho}_{args.surrogate_model}_{args.surrogate_dataset}/bestsurrogatetamodel.pth")
                    self.model = model
        if self.args.taskmode == 'transfer_attack':
            if self.args.source_model_path == None:
                raise NotImplementedError
            else:
                self.model = self.__LoadModel__()        
        if self.args.taskmode == 'transfer_attack_target':
            if self.args.target_model_path == None:
                raise NotImplementedError
            else:
                self.model = self.__LoadModel__()  
                   
        if self.args.taskmode == 'adapt_target':
            if self.args.adapt_target_method =='stdtrain':
                if self.args.target_model_path == None:
                    self.model = self.__GetModel__()
                else:
                    self.model = self.__LoadModel__()         
                          
            elif self.args.adapt_target_method =='finetune': # transfer learn from source model
                if self.args.target_model_path == None:
                    self.model = self.__GetModel__()
                else:
                    self.model = self.__LoadModel__()      
                            
            elif self.args.adapt_target_method =='taskdistill': # transfer learn from source model
                if self.args.target_model_path == None:
                    self.model = self.__GetModel__()
                else:
                    self.model = self.__LoadModel__()    
                self.teacher_model = self.__GetTeacherModel__()
                
            elif self.args.adapt_target_method =='card': # transfer learn from source model
                print("self.args.adapt_target_method:", self.args.adapt_target_method)
                if self.args.target_model_path == None:
                    print('hello, self.args.target_model_path == None')
                    self.model = self.__GetModel__()
                else:
                    self.model = self.__LoadModel__()    
                self.teacher_model = self.__GetTeacherModel__()
                                                        
    def __GetModel__(self):
        
        if self.args.taskmode == 'pretrain_source':
            if self.args.source_model in [
                'vitb16','vitb32','vitl16','vitl32','vith14','convnexttiny','convnextsmall','convnextbase','convnextlarge',
                'convnextv2atto','convnextv2femto','convnextv2pico','convnextv2nano','convnextv2tiny','convnextv2base','convnextv2large','convnextv2huge',
                'clip','blip2']:
                # ----------vit----------                        
                if self.args.source_model == 'vitb16':  
                    """ 
                    B:  Base
                    16: Patch Size --> 16x16
                    32: Patch Size --> 32x32
                    L: Large
                    H: Huge
                    """
                    model = torchmodels.vit_b_16(pretrained=True)             

                if self.args.source_model == 'vitb32':  
                    model = torchmodels.vit_b_32(pretrained=True)          
                    
                if self.args.source_model == 'vitl16':  
                    model = torchmodels.vit_l_16(pretrained=True)
                                        
                if self.args.source_model == 'vitl32':  
                    model = torchmodels.vit_l_32(pretrained=True)

                # ----------convnext----------                    
                if self.args.source_model == 'convnexttiny':  
                    model = torchmodels.convnext_tiny(pretrained=True)
                    
                if self.args.source_model == 'convnextsmall':  
                    model = torchmodels.convnext_small(pretrained=True)

                if self.args.source_model == 'convnextbase':  
                    model = torchmodels.convnext_base(pretrained=True)

                if self.args.source_model == 'convnextlarge':  
                    model = torchmodels.convnext_large(pretrained=True)                            

                # ----------convnextv2----------            
                if self.args.source_model == 'convnextv2atto':  
                    model = convnextv2.convnextv2_atto() 
                    checkpoint = torch.load('result/savemodel/sourcemodel/convnextv2/convnextv2_atto_1k_224_ema.pt')
                    model.load_state_dict(checkpoint['model'])                  

                if self.args.source_model == 'convnextv2femto':  
                    model = convnextv2.convnextv2_femto() 
                    checkpoint = torch.load('result/savemodel/sourcemodel/convnextv2/convnextv2_femto_1k_224_ema.pt')
                    model.load_state_dict(checkpoint['model'])    

                if self.args.source_model == 'convnextv2pico':  
                    model = convnextv2.convnextv2_pico() 
                    checkpoint = torch.load('result/savemodel/sourcemodel/convnextv2/convnextv2_pico_1k_224_ema.pt')
                    model.load_state_dict(checkpoint['model'])   
                    
                if self.args.source_model == 'convnextv2nano':  
                    model = convnextv2.convnextv2_nano() 
                    checkpoint = torch.load('result/savemodel/sourcemodel/convnextv2/convnextv2_nano_1k_224_ema.pt')
                    model.load_state_dict(checkpoint['model'])                        
                                                                                     
                if self.args.source_model == 'convnextv2tiny':  
                    model = convnextv2.convnextv2_tiny()  
                    checkpoint = torch.load('result/savemodel/sourcemodel/convnextv2/convnextv2_tiny_1k_224_ema.pt')
                    model.load_state_dict(checkpoint['model'])   
                                  
                if self.args.source_model == 'convnextv2base':  
                    model = convnextv2.convnextv2_base()
                    checkpoint = torch.load('result/savemodel/sourcemodel/convnextv2/convnextv2_base_1k_224_ema.pt')
                    model.load_state_dict(checkpoint['model'])   
                     
                if self.args.source_model == 'convnextv2large':  
                    model = convnextv2.convnextv2_large()
                    checkpoint = torch.load('result/savemodel/sourcemodel/convnextv2/convnextv2_large_1k_224_ema.pt')
                    model.load_state_dict(checkpoint['model'])   
                    
                if self.args.source_model == 'convnextv2huge':  
                    model = convnextv2.convnextv2_huge()                                        
                    checkpoint = torch.load('result/savemodel/sourcemodel/convnextv2/convnextv2_huge_1k_224_ema.pt')
                    model.load_state_dict(checkpoint['model'])   
                    
                # ----------clip----------                                            
                if self.args.source_model == 'clip':  
                    print("source_model=",self.args.source_model)
                    model = clipmodel.build_model(state_dict='')                    

                # ----------blip2----------                                                                
                if self.args.source_model == 'blip2':  
                    print("source_model=",self.args.source_model)
                    model = blip2model.blip_feature_extractor()
                    # model = blip2model.blip_decoder()
                    
            else:
                raise Exception('please input the valid source model')
            
            # model_savepath = f'result/savemodel/pretrain-{self.args.source_model}.hdf5'
            model_savepath = f'result/savemodel/sourcemodel/pretrain-{self.args.source_model}.hdf5'            
            torch.save(model, model_savepath)

        if self.args.taskmode in ['finetune_surrogate','surrogate_genae']:
            
            if self.args.surrogate_model in ['resnet18','resnet34','resnet50','vgg16', 'inceptionv3','densenet121','swinb','swinbv2']:
                
                if self.args.surrogate_model == 'resnet18':
                    model = torchmodels.resnet18(pretrained=True)  
                if self.args.surrogate_model == 'resnet34':
                    model = torchmodels.resnet34(pretrained=True)  

                if self.args.surrogate_model == 'resnet50':
                    model = torchmodels.resnet50(pretrained=True)  

                if self.args.surrogate_model == 'vgg16':
                    model = torchmodels.vgg16(pretrained=True)  
                    
                if self.args.surrogate_model == 'densenet121':
                    model = torchmodels.densenet121(pretrained=True)  

                if self.args.surrogate_model == 'inceptionv3':
                    model = torchmodels.inception_v3(pretrained=True) 
                    
                if self.args.surrogate_model == 'swinb':
                    model = torchmodels.swin_b(pretrained=True) 
                    
                if self.args.surrogate_model == 'swinbv2':
                    model = torchmodels.swin_v2_b(pretrained=True) 
                
                    
            else:
                raise Exception('please input the valid surrogate model')   
                                 
            model_savepath = f'result/savemodel/surrogatemodel/pretrain-{self.args.surrogate_model}.hdf5'
            torch.save(model, model_savepath)  
                                    
        if self.args.taskmode == 'transfer_attack' or self.args.taskmode =='transfer_attack_target':
            if self.args.surrogate_model in ['resnet18','resnet50','vgg16', 'inceptionv3','densenet121']:
                
                if self.args.surrogate_model == 'resnet18':
                    if self.args.surrogate_dataset in ['imagenet','imagenette']:
                        model = torchmodels.resnet18(pretrained=True)  
                        model_savepath = f'result/savemodel/pretrain-{self.args.surrogate_model}.hdf5'
                        torch.save(model, model_savepath)
                    else:
                        model = torchmodels.resnet18()  
            
                if self.args.surrogate_model == 'resnet50':
                    if self.args.surrogate_dataset in ['imagenet','imagenette']:
                        model = torchmodels.resnet50(pretrained=True)  
                        model_savepath = f'result/savemodel/pretrain-{self.args.surrogate_model}.hdf5'
                        torch.save(model, model_savepath)
                    else:
                        model = torchmodels.resnet50()  
            else:
                raise Exception('please input the valid surrogate model')
            
            model_savepath = f'result/savemodel/pretrain-{self.args.surrogate_model}.hdf5'
            torch.save(model, model_savepath)                    
        
        if self.args.taskmode == 'adapt_target':
            
            if self.args.adapt_target_method in ['stdtrain', 'taskdistill']:
                if self.args.target_model == 'mobilenetv2':
                    model = torchmodels.mobilenet_v2(num_classes=self.args.target_dataset_num_classes)  
                    
                if self.args.target_model == 'mobilenetv3':
                    model = torchmodels.mobilenet_v3_small(num_classes=self.args.target_dataset_num_classes)  
                    
                elif self.args.target_model == 'efficientnet':
                    model = torchmodels.efficientnet_b5(num_classes=self.args.target_dataset_num_classes)  
                    
            elif self.args.adapt_target_method =='finetune': # transfer learn from source model
                model = torch.load(self.args.source_model_path)
                print("target model before finetune = source_model_path:",self.args.source_model_path)
       
            # elif self.args.adapt_target_method =='taskdistill': # transfer learn from source model
            #     if self.args.target_model == 'mobilenetv2':
            #         model = torchmodels.mobilenet_v2(num_classes=self.args.target_dataset_num_classes)  
                    
            #     if self.args.target_model == 'mobilenetv3':
            #         model = torchmodels.mobilenet_v3_small(num_classes=self.args.target_dataset_num_classes)  
                    
            #     elif self.args.target_model == 'efficientnet':
            #         model = torchmodels.efficientnet_b5(num_classes=self.args.target_dataset_num_classes)    
                    
            elif self.args.adapt_target_method =='card': 
                if self.args.target_model == 'mobilenetv2':
                    model = mobilenet_v2(num_classes=self.args.target_dataset_num_classes)  
                    
                if self.args.target_model == 'mobilenetv3':
                    model = mobilenet_v3_small(num_classes=self.args.target_dataset_num_classes)  
                    
                elif self.args.target_model == 'efficientnet':
                    model = efficientnet_b5(num_classes=self.args.target_dataset_num_classes)                          
        return model
    
    def __LoadModel__(self):
        if  self.args.taskmode == 'pretrain_source':
            model = torch.load(self.args.source_model_path)
            print("source_model_path:",self.args.source_model_path)

        elif self.args.taskmode in ['finetune_surrogate','surrogate_genae']:
            if self.args.usingsp:
                if self.args.surrogate_model =='resnet18':
                    model = torchmodels.resnet18()  
                    num_ftrs = model.fc.in_features
                    model.fc = torch.nn.Linear(num_ftrs, 10)
                    model.load_state_dict(torch.load(self.args.surrogate_model_path))
                    print("surrogate_model_path:",self.args.surrogate_model_path)
                elif self.args.surrogate_model == 'densenet121':
                    model = torchmodels.densenet121()  
                    num_ftrs = model.classifier.in_features
                    model.classifier = torch.nn.Linear(num_ftrs, 10) 
                    model.load_state_dict(torch.load(self.args.surrogate_model_path))
                    print("surrogate_model_path:",self.args.surrogate_model_path)
                else:
                    raise NotImplementedError
            
            else:
                model = torch.load(self.args.surrogate_model_path)
                print("surrogate_model_path:",self.args.surrogate_model_path)

        elif self.args.taskmode == 'transfer_attack':
            model = torch.load(self.args.source_model_path)
            print("source_model_path:",self.args.source_model_path)
        elif self.args.taskmode == 'transfer_attack_target':
            model = torch.load(self.args.target_model_path)
            print("source_model_path:",self.args.target_model_path)

        elif self.args.taskmode == 'adapt_target':
            model = torch.load(self.args.target_model_path)
            print("target_model_path:",self.args.target_model_path)
                
        return model
    
    def __GetTeacherModel__(self):
        if self.args.taskmode == 'adapt_target':
            teacher_model = torch.load(self.args.source_model_path)
            print("teacher model of target model = source_model_path:",self.args.source_model_path)
        else:
            raise Exception(f"Please input the valid teacher model of {self.args.target_model}")
        return teacher_model
              
    def train(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")            

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        # writer = SummaryWriter(logdir=self.args.save_path)  # 创建一个SummaryWriter对象来写入TensorBoard
        early_stopping = EarlyStopping(save_path=self.args.save_path, patience=50) 
        scheduler = CosineAnnealingLR(self.optimizer, T_max=10)  # T_max 是学习率下降的周期长度


        train_losses = []
        test_losses = []
        top1_accs = []
        top5_accs = []
        epoch_times = []
        
        for epoch_index in range(self.args.n_epochs):
            start_time = time.time()  # 记录每个epoch的开始时间
            self.model.train()   
            running_loss = 0.0
            
            for i, (inputs, labels) in enumerate(self.trainloader, 0):
                inputs, labels = inputs.to(device), labels.to(device)
                self.optimizer.zero_grad() 
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            
            end_time = time.time()  # 记录每个epoch的结束时间
            epoch_time = end_time - start_time  # 计算每个epoch的训练时间
            
            train_loss = running_loss / len(self.trainloader)

            self.model.eval()  
            top1_acc, top5_acc, test_loss, evaluate_time = evaluate_model(self.model, self.testloader, self.criterion)
        
            print(f'[{epoch_index:04d} epoch] Epoch Time: {epoch_time:.2f} seconds, Train Loss: {train_loss:.3f}, Test Loss: {test_loss:.3f}, Top1 Accuracy: {top1_acc:.3f}, Top5 Accuracy: {top5_acc:.3f}')

            # writer.add_scalar('Train Loss', train_loss, epoch_index)
            # writer.add_scalar('Test Loss', test_loss, epoch_index)
            # writer.add_scalar('Top1 Accuracy', top1_acc, epoch_index)
            # writer.add_scalar('Top5 Accuracy', top5_acc, epoch_index)
            # writer.add_scalar('Epoch Time', epoch_time, epoch_index)
 
            line_chart('epoch_index', 'train_loss', epoch_index, train_loss, self.args.save_path)
            line_chart('epoch_index', 'test_loss', epoch_index, test_loss, self.args.save_path)
            line_chart('epoch_index', 'top1_acc', epoch_index, top1_acc, self.args.save_path)
            line_chart('epoch_index', 'top5_acc', epoch_index, top5_acc, self.args.save_path)
            line_chart('epoch_index', 'epoch_time', epoch_index, epoch_time, self.args.save_path)

            
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            top1_accs.append(top1_acc)
            top5_accs.append(top5_acc)
            epoch_times.append(epoch_time)
           
            early_stopping(test_loss, self.model)
            if early_stopping.early_stop == True:
                print("Early stopping")
                model_savepath = f'{self.args.save_path}/{self.args.surrogate_model}-{self.args.surrogate_dataset}-stdtrain-epoch-{i:04d}-top1acc-{top1_acc:.3f}-top5acc-{top5_acc:.3f}.hdf5'
                torch.save(self.model, model_savepath)
                break 
            
            scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch_index}: current learning rate = {current_lr}')
                                  
        plt.figure(figsize=(4, 3.5))
        plt.plot(range(1, self.args.n_epochs+1), train_losses, marker='o',label='Train Loss')
        plt.xlabel('Epoch', fontsize=10)
        plt.ylabel('Loss', fontsize=10)
        plt.title('Train Loss', fontsize=10)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{self.args.save_path}/Train-Loss.png')
        plt.show()
        plt.close()

        plt.figure(figsize=(4, 3.5))        
        plt.plot(range(1, self.args.n_epochs+1), test_losses, marker='o',label='Test Loss')
        plt.xlabel('Epoch', fontsize=10)
        plt.ylabel('Loss', fontsize=10)
        plt.title('Test Loss', fontsize=10)
        plt.legend()
        plt.xticks(range(1, self.args.n_epochs+1))
        plt.tight_layout()
        plt.savefig(f'{self.args.save_path}/Test-Loss.png')
        plt.show()
        plt.close()
        
        plt.figure(figsize=(4, 3.5))
        plt.plot(range(1, self.args.n_epochs+1), top1_accs, marker='o',label='Top1 Accuracy')
        plt.xlabel('Epoch', fontsize=10)
        plt.ylabel('Accuracy', fontsize=10)
        plt.title('Top1 Accuracy', fontsize=10)
        plt.legend()
        plt.xticks(range(1, self.args.n_epochs+1))
        plt.tight_layout()
        plt.savefig(f'{self.args.save_path}/Top1-Accuracy.png')
        plt.show()
        plt.close()
        
        plt.figure(figsize=(4, 3.5))
        plt.plot(range(1, self.args.n_epochs+1), top5_accs, marker='o',label='Top5 Accuracy')
        plt.xlabel('Epoch', fontsize=10)
        plt.ylabel('Accuracy', fontsize=10)
        plt.title('Top5 Accuracy', fontsize=10)
        plt.legend()
        plt.xticks(range(1, self.args.n_epochs+1))
        plt.tight_layout()
        plt.savefig(f'{self.args.save_path}/Top5-Accuracy.png')
        plt.show()
        plt.close()
        
        plt.figure(figsize=(4, 3.5))
        plt.plot(range(1, self.args.n_epochs+1), epoch_times, marker='o',label='Epoch Time')
        plt.xlabel('Epoch', fontsize=10)
        plt.ylabel('Time (Seconds)', fontsize=10)
        plt.title('Time Cost per Epoch', fontsize=10)
        plt.legend()
        plt.xticks(range(1, self.args.n_epochs+1))
        plt.tight_layout()
        plt.savefig(f'{self.args.save_path}/Epoch-Time.png')
        plt.show()
        plt.close()
                            
        print('Finished Training')
 

