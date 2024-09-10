
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
from utils.parseargs import parse_argu
from utils.eval import evaluate_model
import os
from models.model import TARFAModel
from dataset.datapro import *
from train.finetune import finetune_surrogate, finetune_source
from train.stdtrain import stdtrain
from attack.genae import genae
from train.distill import taskdistill
import torchvision.models as torchmodels
from train.card import card

base_path = 'root_path'                 

if __name__ == '__main__':
    
    print("======== Maggie is the best! ========\n")
    
    if torch.cuda.is_available():
        print('Torch cuda is available')
        print(f'Total number of GPUs available: {torch.cuda.device_count()}')
        print("Current device:", torch.cuda.current_device())
        print(f'Total number of CPUs available: {os.cpu_count()}')
        """ 
        Total number of GPUs available: 3
        Total number of CPUs available: 48
        """
    else:
        print("Torch cuda is not available.")
        print(f'Total number of CPUs available: {os.cpu_count()}')
        
        """
        Torch cuda is not available.
        Total number of CPUs available: 48
        """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    args =parse_argu()
    print('args=%s' % args)          
    print('args.save_path=%s' % args.save_path)
 
    device_ids = []
    if args.n_gpus == 2:
        device_ids = [1,2]
    elif args.n_gpus == 3:
        device_ids = [0,1,2]
    print(f"Using GPUs: {device_ids}")

    if args.taskmode == 'pretrain_source':
        print(f"taskmode: {args.taskmode}")
        SourceModel = TARFAModel(args)
        
        _, _, source_test_loader = get_source_dataloader(args)
        SourceModel.testloader = source_test_loader

        if args.n_gpus > 1:
            SourceModel.model = torch.nn.DataParallel(SourceModel.model, device_ids=device_ids)
                             
        print(f"Evaluate the pretrain source model {args.source_model} top1 and top5 accuracy on {args.source_dataset} test set ...")
        metrics = evaluate_model(SourceModel.model, SourceModel.testloader, SourceModel.criterion)
        print(f'Top1 Accuracy on {args.source_dataset} test set: {metrics[0] * 100:.3f}%')
        print(f'Top5 Accuracy on {args.source_dataset} test set: {metrics[1] * 100:.3f}%')
        print(f'Average Loss on {args.source_dataset} test set: {metrics[2]:.3f}')
        print(f"Evaluation Time: {metrics[3]:.3f} seconds")    
    
    if args.taskmode == 'finetune_surrogate':
        print(f"taskmode: {args.taskmode}")
        SurrogateModel = TARFAModel(args)

        surrogate_train_loader, surrogate_val_loader, surrogate_test_loader = get_surrogate_dataloader(args=args, clean=True)
    
        SurrogateModel.trainloader = surrogate_train_loader
        SurrogateModel.valloader = surrogate_val_loader
        SurrogateModel.testloader = surrogate_test_loader
        
        SurrogateModel.num_classes = len(set([label for _, label in SurrogateModel.testloader.dataset]))
        print("SurrogateModel.num_classes:", SurrogateModel.num_classes)
        """
        args.surrogate_dataset from imagenette
        SurrogateModel.num_classes: 10       
        
        args.surrogate_dataset from tiny_imagenet
        SurrogateModel.num_classes: 200
        """            
        
        if args.n_gpus > 1:
            SourceModel.model = torch.nn.DataParallel(SourceModel.model, device_ids=device_ids)      

        finetune_time = finetune_surrogate(SurrogateModel.model, SurrogateModel.trainloader, SurrogateModel.valloader, SurrogateModel.criterion, SurrogateModel.num_classes, args)
            
        print(f"Evaluate the finetuned surrogate model {args.surrogate_model} top1 and top5 accuracy on {args.surrogate_dataset} test set ...")        
        
        metrics = evaluate_model(SurrogateModel.model, SurrogateModel.testloader, SurrogateModel.criterion)
        print(f'Top1 Accuracy on {args.surrogate_dataset} test set: {metrics[0] * 100:.3f}%')
        print(f'Top5 Accuracy on {args.surrogate_dataset} test set: {metrics[1] * 100:.3f}%')
        print(f'Average Loss on {args.surrogate_dataset} test set: {metrics[2]:.3f}')
        print(f"Evaluation Time: {metrics[3]:.3f} seconds") 
        print(f'Finetune time on {args.surrogate_dataset} training set: {finetune_time:.3f}')

        os.makedirs(f'result/savemodel/surrogatemodel/{args.surrogate_dataset}/{args.surrogate_model}', exist_ok=True)
        model_savepath = f'result/savemodel/surrogatemodel/{args.surrogate_dataset}/{args.surrogate_model}/finetuned-{args.surrogate_model}-{args.surrogate_dataset}-{args.finetune_mode}-top1acc-{metrics[0] * 100:.3f}-top5acc-{metrics[1] * 100:.3f}.hdf5'
        torch.save(SurrogateModel.model, model_savepath)
                                         
    if args.taskmode == 'surrogate_genae':
        
        print(f"taskmode: {args.taskmode}")
       
        SurrogateModel = TARFAModel(args)
        surrogate_train_loader, surrogate_val_loader, surrogate_test_loader = get_surrogate_dataloader(args=args, clean=True)
    
        SurrogateModel.trainloader = surrogate_train_loader
        SurrogateModel.valloader = surrogate_val_loader
        SurrogateModel.testloader = surrogate_test_loader
        
        SurrogateModel.num_classes = len(set([label for _, label in SurrogateModel.testloader.dataset]))
        print("SurrogateModel.num_classes:", SurrogateModel.num_classes)
        """
        args.surrogate_dataset from imagenette
        SurrogateModel.num_classes: 10       
        
        args.surrogate_dataset from tiny_imagenet
        SurrogateModel.num_classes: 200
        """            
        
        if args.n_gpus > 1:
            SourceModel.model = torch.nn.DataParallel(SourceModel.model, device_ids=device_ids)   

        #---------test surrogate model on clean surrogate dataset---------
        metrics = evaluate_model(SurrogateModel.model, SurrogateModel.testloader, SurrogateModel.criterion)
        print(f'Top1 Accuracy on {args.surrogate_dataset} test set: {metrics[0] * 100:.3f}%')
        print(f'Top5 Accuracy on {args.surrogate_dataset} test set: {metrics[1] * 100:.3f}%')
        print(f'Average Loss on {args.surrogate_dataset} test set: {metrics[2]:.3f}')
        print(f"Evaluation Time: {metrics[3]:.3f} seconds") 
        
        #---------generate adv surrogate dataset based surrogate model---------        
        print(f"surrogate-{args.surrogate_model}-generate {args.attack_type}-{args.ae_eps}-{args.ae_step}-{args.ae_ite} adversarial-{args.surrogate_dataset}-examples")
        
        adv_testloader, genrate_time = genae(SurrogateModel.model, SurrogateModel.testloader, SurrogateModel.criterion, SurrogateModel.num_classes, args)

        #---------test surrogate model on adv surrogate dataset---------        
        print(f"Evaluate the trained surrogate model {args.surrogate_model} top1 and top5 accuracy on {args.attack_type} adversarial {args.surrogate_dataset} ...")
        metrics = evaluate_model(SurrogateModel.model, adv_testloader, SurrogateModel.criterion)
        print(f'Top1 Accuracy on {args.surrogate_dataset} adv test set: {metrics[0] * 100:.3f}%')
        print(f'Top5 Accuracy on {args.surrogate_dataset} adv test set: {metrics[1] * 100:.3f}%')
        print(f'Average Loss on {args.surrogate_dataset} adv test set: {metrics[2]:.3f}')
        print(f"Evaluation Time: {metrics[3]:.3f} seconds")      
        print(f'Generate time on {args.surrogate_dataset} test set: {genrate_time:.3f}')
        
        #---------save adv surrogate dataset---------                
        all_images = []
        all_labels = []
        for images, labels in adv_testloader:
            all_images.append(images)
            all_labels.append(labels)
        all_images = torch.cat(all_images, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        adv_save_path = base_path + f'/data/TARFA-Result/surrogate_genae/{args.surrogate_model}/{args.surrogate_dataset}/'
        os.makedirs(adv_save_path, exist_ok=True)
        if not args.usingsp:
            torch.save(all_images, f'{adv_save_path}/{args.surrogate_model}-{args.surrogate_dataset}-{args.attack_type}-{args.ae_eps}-{args.ae_step}-{args.ae_ite}-top1acc-{metrics[0] * 100:.3f}-top5acc-{metrics[1] * 100:.3f}-adv-test-images.pt')
            torch.save(all_labels, f'{adv_save_path}/{args.surrogate_model}-{args.surrogate_dataset}-{args.attack_type}-{args.ae_eps}-{args.ae_step}-{args.ae_ite}-top1acc-{metrics[0] * 100:.3f}-top5acc-{metrics[1] * 100:.3f}-adv-test-labels.pt')
        else:
            torch.save(all_images, f'{adv_save_path}/{args.surrogate_model}_sp-{args.surrogate_dataset}-{args.attack_type}-{args.ae_eps}-{args.ae_step}-{args.ae_ite}-top1acc-{metrics[0] * 100:.3f}-top5acc-{metrics[1] * 100:.3f}-adv-test-images.pt')
            torch.save(all_labels, f'{adv_save_path}/{args.surrogate_model}_sp-{args.surrogate_dataset}-{args.attack_type}-{args.ae_eps}-{args.ae_step}-{args.ae_ite}-top1acc-{metrics[0] * 100:.3f}-top5acc-{metrics[1] * 100:.3f}-adv-test-labels.pt')
    
    if args.taskmode == 'transfer_attack':
        print(f"taskmode: {args.taskmode}")
        SourceModel = TARFAModel(args)
        
        if args.n_gpus > 1:
            SourceModel.model = torch.nn.DataParallel(SourceModel.model, device_ids=device_ids)

        #===========================  
        if args.usingsp:
            if args.surrogate_model =='resnet18':
                surrogate_model = torchmodels.resnet18()  
                num_ftrs = surrogate_model.fc.in_features
                surrogate_model.fc = torch.nn.Linear(num_ftrs, 10)
                surrogate_model.load_state_dict(torch.load(args.surrogate_model_path))
                print("surrogate_model_path:",args.surrogate_model_path)
            elif args.surrogate_model == 'densenet121':
                surrogate_model = torchmodels.densenet121()  
                num_ftrs = surrogate_model.classifier.in_features
                surrogate_model.classifier = torch.nn.Linear(num_ftrs, 10) 
                surrogate_model.load_state_dict(torch.load(args.surrogate_model_path))
                print("surrogate_model_path:",args.surrogate_model_path)
            else:
                raise NotImplementedError
        else:
            surrogate_model = torch.load(args.surrogate_model_path)
        print("surrogate_model_path:",args.surrogate_model_path)
        
        print(f"Evaluate the args.usingsp={args.usingsp} surrogate model {args.surrogate_model} top1 and top5 accuracy on adversarial {args.surrogate_dataset} examples ...")
        _, _, ori_surrogate_test_loader_adv = get_surrogate_dataloader(args=args, adv=True)
        print("len(ori_surrogate_test_loader_adv.dataset):",len(ori_surrogate_test_loader_adv.dataset))  
        metrics = evaluate_model(surrogate_model, ori_surrogate_test_loader_adv, torch.nn.CrossEntropyLoss())
        print(f'{args.surrogate_model} Top1 Accuracy on {args.surrogate_dataset} {args.attack_type} adv test set: {metrics[0] * 100:.3f}%')
        print(f'{args.surrogate_model} Top5 Accuracy on {args.surrogate_dataset} {args.attack_type} adv test set: {metrics[1] * 100:.3f}%')
        print(f'{args.surrogate_model} Average Loss on {args.surrogate_dataset} {args.attack_type} adv test set: {metrics[2]:.3f}')
        print(f"{args.surrogate_model} Evaluation Time: {metrics[3]:.3f} seconds")       
        #===========================

    
    
        #===========================
        #---------important---------
        # """
        # Hi Link, All of the get_filter_source_dataloader function should not be used, because this function is used to sampled data from the original ImageNet dataset. I developed this function for previously test the impact of diffence between the released imagenette/released imagewoof and my created imagenette (imagenet-10-in) and imagewoof (imagenet-10-iw). 
        
        # we just need to use get_align_surrogate_dataloader to align the index in imagenette/imagewoof with ImageNet.
        # """
        if args.surrogate_dataset == 'imagenet-200':
            _, _, source_test_loader = get_source_dataloader(args)
            ImageNet_200_test_loader = get_filter_source_dataloader(source_test_loader, args, 'tiny_imagenet')
            print("len(ImageNet_200_test_loader.dataset):",len(ImageNet_200_test_loader.dataset))
            """ 
            len(ImageNet_200_test_loader.dataset): 10000
            """
            SourceModel.testloader = ImageNet_200_test_loader
        # elif args.surrogate_dataset == 'imagenette':
        elif args.surrogate_dataset == 'imagenet-10-in':
            _, _, source_test_loader = get_source_dataloader(args)
            ImageNet_10_IN_test_loader = get_filter_source_dataloader(source_test_loader, args, 'imagenette')
            print("len(ImageNet_10_IN_test_loader.dataset):",len(ImageNet_10_IN_test_loader.dataset))
            """ 
            len(ImageNet_10_IN_test_loader.dataset): 500
            """
            SourceModel.testloader = ImageNet_10_IN_test_loader     
        # elif args.surrogate_dataset == 'imagewoof':
        elif args.surrogate_dataset == 'imagenet-10-iw':
            _, _, source_test_loader = get_source_dataloader(args)
            ImageNet_10_IW_test_loader = get_filter_source_dataloader(source_test_loader, args, 'imagewoof')
            print("len(ImageNet_10_IW_test_loader.dataset):",len(ImageNet_10_IW_test_loader.dataset))
            """ 
            len(ImageNet_10_IW_test_loader.dataset): 500
            """
            SourceModel.testloader = ImageNet_10_IW_test_loader            
        else:   # this should be used
            surrogate_train_loader_cle, surrogate_val_loader_cle, surrogate_test_loader_cle = get_align_surrogate_dataloader(args=args, clean=True)
            print("len(surrogate_test_loader_cle.dataset):",len(surrogate_test_loader_cle.dataset))
            SourceModel.testloader = surrogate_test_loader_cle

        print(f"Evaluate the pretrained source model {args.source_model} top1 and top5 accuracy on clean {args.surrogate_dataset} examples ...")
        print("SampleNum = len(SourceModel.testloader.dataset):",len(SourceModel.testloader.dataset))
        metrics = evaluate_model(SourceModel.model, SourceModel.testloader, SourceModel.criterion)
        print(f'{args.source_model} Top1 Accuracy on {args.surrogate_dataset} clean test set: {metrics[0] * 100:.3f}%')
        print(f'{args.source_model} Top5 Accuracy on {args.surrogate_dataset} clean test set: {metrics[1] * 100:.3f}%')
        print(f'{args.source_model} Average Loss on {args.surrogate_dataset} clean test set: {metrics[2]:.3f}')
        print(f"{args.source_model} Evaluation Time: {metrics[3]:.3f} seconds")     
        #===========================

        #===========================
        _, _, surrogate_test_loader_adv = get_align_surrogate_dataloader(args=args, adv=True)
        print("len(surrogate_test_loader_adv.dataset):",len(surrogate_test_loader_adv.dataset))            
        SourceModel.advtestloader = surrogate_test_loader_adv
        print(f"Evaluate the pretrained source model {args.source_model} top1 and top5 accuracy on adversarial {args.surrogate_dataset} examples ...")
        print("SampleNum = len(SourceModel.advtestloader.dataset):",len(SourceModel.advtestloader.dataset))
        metrics = evaluate_model(SourceModel.model, SourceModel.advtestloader, SourceModel.criterion)
        print(f'{args.source_model} Top1 Accuracy on {args.surrogate_dataset} {args.attack_type} adv test set (args.all={args.all}): {metrics[0] * 100:.3f}%')
        print(f'{args.source_model} Top5 Accuracy on {args.surrogate_dataset} {args.attack_type} adv test set (args.all={args.all}): {metrics[1] * 100:.3f}%')
        print(f'{args.source_model} Average Loss on {args.surrogate_dataset} {args.attack_type} adv test set (args.all={args.all}): {metrics[2]:.3f}')
        print(f"{args.source_model} Evaluation Time: {metrics[3]:.3f} seconds")       
        #===========================

    if args.taskmode == 'transfer_attack_target':
        print(f"taskmode: {args.taskmode}")
        TargetModel = TARFAModel(args)
        
        if args.n_gpus > 1:
            TargetModel.model = torch.nn.DataParallel(TargetModel.model, device_ids=device_ids)


        #============================
        if args.usingsp:
            if args.surrogate_model =='resnet18':
                surrogate_model = torchmodels.resnet18()  
                num_ftrs = surrogate_model.fc.in_features
                surrogate_model.fc = torch.nn.Linear(num_ftrs, 10)
                surrogate_model.load_state_dict(torch.load(args.surrogate_model_path))
                print("surrogate_model_path:",args.surrogate_model_path)
            elif args.surrogate_model == 'densenet121':
                surrogate_model = torchmodels.densenet121()  
                num_ftrs = surrogate_model.classifier.in_features
                surrogate_model.classifier = torch.nn.Linear(num_ftrs, 10) 
                surrogate_model.load_state_dict(torch.load(args.surrogate_model_path))
                print("surrogate_model_path:",args.surrogate_model_path)
            else:
                raise NotImplementedError
        else:
            surrogate_model = torch.load(args.surrogate_model_path)
            
        print("surrogate_model_path:",args.surrogate_model_path)
        print(f"Evaluate the args.usingsp={args.usingsp} surrogate model {args.surrogate_model} top1 and top5 accuracy on adversarial {args.surrogate_dataset} examples ...")
        _, _, ori_surrogate_test_loader_adv = get_surrogate_dataloader(args=args, adv=True)
        print("len(ori_surrogate_test_loader_adv.dataset):",len(ori_surrogate_test_loader_adv.dataset))  
        
        metrics = evaluate_model(surrogate_model, ori_surrogate_test_loader_adv, torch.nn.CrossEntropyLoss())
        #---------maggie add comment: according to code, actual test set here is all not cutted
        print(f'{args.surrogate_model} Top1 Accuracy on {args.surrogate_dataset} {args.attack_type} adv test set (cutted): {metrics[0] * 100:.3f}%')
        print(f'{args.surrogate_model} Top5 Accuracy on {args.surrogate_dataset} {args.attack_type} adv test set (cutted): {metrics[1] * 100:.3f}%')
        print(f'{args.surrogate_model} Average Loss on {args.surrogate_dataset} {args.attack_type} adv test set (cutted): {metrics[2]:.3f}')
        print(f"{args.surrogate_model} Evaluation Time: {metrics[3]:.3f} seconds")       
        #============================


        


        #============================        
        # print(f"Evaluate the pretrained source model {args.target_model} top1 and top5 accuracy on adversarial {args.surrogate_dataset} examples ...")
        print(f"Evaluate the trained-well target model {args.target_model} top1 and top5 accuracy on adversarial {args.target_dataset} examples ...")            
        _, _, surrogate_test_loader_adv = get_align_surrogate_dataloader(args=args, adv=True)
        print("len(surrogate_test_loader_adv.dataset):",len(surrogate_test_loader_adv.dataset))            
        TargetModel.advtestloader = surrogate_test_loader_adv
        print("SampleNum = len(TargetModel.advtestloader.dataset):",len(TargetModel.advtestloader.dataset))
        metrics = evaluate_model(TargetModel.model, TargetModel.advtestloader, TargetModel.criterion)
        print(f'{args.target_model} Top1 Accuracy on {args.surrogate_dataset} {args.attack_type} adv test set (cutted): {metrics[0] * 100:.3f}%')
        print(f'{args.target_model} Top5 Accuracy on {args.surrogate_dataset} {args.attack_type} adv test set (cutted): {metrics[1] * 100:.3f}%')
        print(f'{args.target_model} Average Loss on {args.surrogate_dataset} {args.attack_type} adv test set (cutted): {metrics[2]:.3f}')
        print(f"{args.target_model} Evaluation Time: {metrics[3]:.3f} seconds")         
        #============================
        
        #============================
        # print(f"Evaluate the pretrained source model {args.source_model} top1 and top5 accuracy on clean {args.target_dataset} examples ...")

        # if args.surrogate_dataset == 'imagenette':
        #     _, _, target_test_loader = get_target_dataloader(args)
        #     ImageNet_10_IN_test_loader = get_filter_source_dataloader(target_test_loader, args, 'imagenette')
        #     print("len(ImageNet_10_IN_test_loader.dataset):",len(ImageNet_10_IN_test_loader.dataset))
        #     """ 
        #     len(ImageNet_10_IN_test_loader.dataset): 500
        #     """
        #     TargetModel.testloader = ImageNet_10_IN_test_loader            
        # elif args.surrogate_dataset == 'imagewoof':
        #     _, _, target_test_loader = get_target_dataloader(args)
        #     ImageNet_10_IW_test_loader = get_filter_source_dataloader(target_test_loader, args, 'imagewoof')
        #     print("len(ImageNet_10_IW_test_loader.dataset):",len(ImageNet_10_IW_test_loader.dataset))
        #     """ 
        #     len(ImageNet_10_IW_test_loader.dataset): 500
        #     """
        #     TargetModel.testloader = ImageNet_10_IW_test_loader     













        # if args.attack_type in ['fgsm']:
        #     if args.surrogate_dataset in ['imagenette', 'imagewoof']:
        #         _, _, surrogate_test_loader_cle = get_align_surrogate_dataloader(args=args, clean=True) # here need to map index in imagetted to target dataset(CIFAR10-CIFAR100)
        #         print("len(surrogate_test_loader_cle.dataset):",len(surrogate_test_loader_cle.dataset))
        #         TargetModel.testloader = surrogate_test_loader_cle
                
        #     print(f"Evaluate the trained-well target model {args.target_model} top1 and top5 accuracy on  cutted {args.surrogate_dataset} clean examples ...")            
        #     print("SampleNum = len(TargetModel.testloader.dataset):",len(TargetModel.testloader.dataset))
        #     metrics = evaluate_model(TargetModel.model, TargetModel.testloader, TargetModel.criterion)
        #     print(f'{args.target_model} Top1 Accuracy on {args.surrogate_dataset} clean test set (cutted): {metrics[0] * 100:.3f}%')
        #     print(f'{args.target_model} Top5 Accuracy on {args.surrogate_dataset} clean test set (cutted): {metrics[1] * 100:.3f}%')
        #     print(f'{args.target_model} Average Loss on {args.surrogate_dataset} clean test set (cutted): {metrics[2]:.3f}')
        #     print(f"{args.target_model} Evaluation Time: {metrics[3]:.3f} seconds")     
        #     #============================
    
    if args.taskmode == 'adapt_target':
        print(f">>>>>>>>>>>>>> taskmode: {args.taskmode} >>>>>>>>>>>>>> ")
        
        target_train_loader, target_val_loader, target_test_loader = get_target_dataloader(args=args, clean=True)
        
        target_dataset_num_classes = len(set(target_test_loader.dataset.targets))
        args.target_dataset_num_classes = target_dataset_num_classes
        print("args.target_dataset_num_classes:", args.target_dataset_num_classes)
                
        print(f">>>>>>>>>>>>>> initialize target-domain {args.target_model}>>>>>>>>>>>>>> ")
        TargetModel = TARFAModel(args)
        TargetModel.num_classes = target_dataset_num_classes
        
        if args.n_gpus > 1:
            TargetModel.model = torch.nn.DataParallel(TargetModel.model, device_ids=device_ids)
            if args.adapt_target_method in ['taskdistill','card']: 
                TargetModel.teacher_model = torch.nn.DataParallel(TargetModel.teacher_model, device_ids=device_ids)

        print(f">>>>>>>>>>>>>> load target-domain {args.target_dataset}>>>>>>>>>>>>>> ")       
        TargetModel.trainloader = target_train_loader
        TargetModel.valloader = target_val_loader
        TargetModel.testloader = target_test_loader
        print("len(TargetModel.trainloader.dataset):", len(TargetModel.trainloader.dataset))
 
 
        print(f">>>>>>>>>>>>>> start train target-domain {args.target_model}>>>>>>>>>>>>>> ")               
        if args.adapt_target_method =='stdtrain':
            target_train_time = stdtrain(TargetModel.model, TargetModel.trainloader, TargetModel.valloader, TargetModel.criterion, TargetModel.num_classes, args)
        
        elif args.adapt_target_method =='finetune':
            target_train_time = finetune_source(TargetModel.model, TargetModel.trainloader, TargetModel.valloader, TargetModel.criterion, TargetModel.num_classes, args)

        elif args.adapt_target_method =='taskdistill':             
            target_train_time = taskdistill(TargetModel.model, TargetModel.teacher_model, TargetModel.trainloader, TargetModel.valloader, TargetModel.criterion, TargetModel.num_classes, args)

        elif args.adapt_target_method =='card':
            print("args.adapt_target_method:",args.adapt_target_method)
            args.is_sample = True
            # target_train_time = card(TargetModel.model, TargetModel.teacher_model, TargetModel.trainloader, TargetModel.valloader, TargetModel.criterion, TargetModel.num_classes, device,args)

            target_train_time = card(TargetModel.model, TargetModel.teacher_model, TargetModel.trainloader, TargetModel.testloader, TargetModel.criterion, TargetModel.num_classes, device,args)


        print(f">>>>>>>>>>>>>> finish train target-domain {args.target_model}>>>>>>>>>>>>>> ")                           
        print(f"Evaluate the {args.adapt_target_method} target model {args.target_model} top1 and top5 accuracy on {args.target_dataset} test set ...")                
        metrics = evaluate_model(TargetModel.model, TargetModel.testloader, TargetModel.criterion)
        print(f'{args.adapt_target_method} target model {args.target_model} Top1 Accuracy on {args.target_dataset} test set: {metrics[0] * 100:.3f}%')
        print(f'{args.adapt_target_method} target model {args.target_model} Top5 Accuracy on {args.target_dataset} test set: {metrics[1] * 100:.3f}%')
        print(f'{args.adapt_target_method} target model {args.target_model} Average Test Loss on {args.target_dataset} test set: {metrics[2]:.3f}')
        print(f"{args.adapt_target_method} target model {args.target_model} Evaluation Time: {metrics[3]:.3f} seconds") 
        print(f'{args.adapt_target_method} time on {args.target_dataset} training set: {target_train_time:.3f}')

        os.makedirs(f'result/savemodel/targetmodel/{args.target_dataset}/{args.target_model}/{args.adapt_target_method}', exist_ok=True)
        model_savepath = f'result/savemodel/targetmodel/{args.target_dataset}/{args.target_model}/{args.adapt_target_method}-{args.target_model}-{args.target_dataset}-top1acc-{metrics[0] * 100:.3f}-top5acc-{metrics[1] * 100:.3f}.hdf5'
        torch.save(TargetModel.model, model_savepath)        
           
    print("\n")