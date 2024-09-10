import torch
from utils.ealystop import EarlyStopping
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
from utils.eval import evaluate_model
from utils.tensorboarddraw import line_chart
import matplotlib.pyplot as plt
from utils.draw import drawplt_line
 
class MappingLayer(torch.nn.Module):
    def __init__(self, num_teacher_features, num_student_features):
        super(MappingLayer, self).__init__()
        self.mapping = torch.nn.Linear(num_teacher_features, num_student_features)

    def forward(self, x):
        return self.mapping(x)
    
    
def taskdistill(student_model, teacher_model, trainloader, valloader, criterion, num_classes, args):
    criterion_kl = torch.nn.KLDivLoss(reduction='batchmean')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
    student_model = student_model.to(device)  
    teacher_model = teacher_model.to(device) 
    
    num_teacher_classes = 1000
    if num_classes == num_teacher_classes:
        optimizer = torch.optim.SGD(student_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)    
    else:
        print("-----output_align_layer working-----")
        output_align_layer = MappingLayer(num_teacher_classes, num_classes)
        output_align_layer = output_align_layer.to(device)  
        optimizer = torch.optim.SGD(list(student_model.parameters()) + list(output_align_layer.parameters()), lr=0.01, momentum=0.9, weight_decay=5e-4)    
        
    early_stopping = EarlyStopping(save_path=args.save_path, patience=50) 
    scheduler = CosineAnnealingLR(optimizer, T_max=args.n_epochs)  # T_max 是学习率下降的周期长度

    train_losses = []
    val_losses = []
    top1_accs = []
    top5_accs = []
    epoch_times = []
    
    for epoch_index in range(args.n_epochs):
        start_time = time.time()   
        teacher_model.eval()
        student_model.train()   
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.no_grad():
                teacher_outputs_logits = teacher_model(inputs)
                # print("teacher_outputs_logits.shape:",teacher_outputs_logits.shape)
            
            student_outputs_logits = student_model(inputs)
            # print("student_outputs_logits.shape:",student_outputs_logits.shape)

            if teacher_outputs_logits.shape[1] == student_outputs_logits.shape[1]:
                teacher_outputs_logits = teacher_outputs_logits
            else:
                teacher_outputs_logits = output_align_layer(teacher_outputs_logits)

            # distillation loss
            teacher_outputs_soft = torch.nn.functional.softmax(teacher_outputs_logits / args.distill_temperature, dim=1)
            # print("teacher_outputs_soft.shape:",teacher_outputs_soft.shape)
            student_outputs_soft = torch.nn.functional.softmax(student_outputs_logits / args.distill_temperature, dim=1)
            # print("student_outputs_soft.shape:",student_outputs_soft.shape)

            soft_loss = criterion_kl(student_outputs_soft, teacher_outputs_soft) * (args.distill_temperature ** 2)
            hard_loss = criterion(student_outputs_logits, labels)     
            loss = args.distill_lambda * soft_loss + (1 - args.distill_lambda) * hard_loss
            
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        end_time = time.time()   
        epoch_time = end_time - start_time   
        train_loss = running_loss / len(trainloader)
        student_model.eval()  
        top1_acc, top5_acc, val_loss, evaluate_time = evaluate_model(student_model, valloader, criterion)
        
    
        print(f'[{epoch_index:04d} epoch] Epoch Time: {epoch_time:.2f} seconds, Train Loss: {train_loss:.3f}, Validation Loss: {val_loss:.3f}, Top1 Accuracy: {top1_acc:.3f}, Top5 Accuracy: {top5_acc:.3f}')

        line_chart('epoch_index', 'train_loss', epoch_index, train_loss, args.save_path)
        line_chart('epoch_index', 'val_loss', epoch_index, val_loss, args.save_path)
        line_chart('epoch_index', 'val_top1_acc', epoch_index, top1_acc, args.save_path)
        line_chart('epoch_index', 'val_top5_acc', epoch_index, top5_acc, args.save_path)
        line_chart('epoch_index', 'epoch_time', epoch_index, epoch_time, args.save_path)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        top1_accs.append(top1_acc)
        top5_accs.append(top5_acc)
        epoch_times.append(epoch_time)
        
        early_stopping(val_loss, student_model)
        if early_stopping.early_stop == True:
            print("Early stopping")
            model_savepath = f'{args.save_path}/{args.target_model}-{args.target_dataset}-stdtrain-epoch-{i:04d}-top1acc-{top1_acc:.3f}-top5acc-{top5_acc:.3f}.hdf5'
            torch.save(student_model, model_savepath)
            break 
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch_index}: current learning rate = {current_lr}')
            
                        
    drawplt_line(x=range(1, args.n_epochs+1), y=train_losses, xlabel='Epoch', ylabel='Training Loss', title=f'taskdistill-{args.source_model}-to-{args.target_model}-{args.target_dataset}', savepth=args.save_path)
    drawplt_line(x=range(1, args.n_epochs+1), y=val_losses, xlabel='Epoch', ylabel='Validation Loss', title=f'taskdistill-{args.source_model}-to-{args.target_model}-{args.target_dataset}', savepth=args.save_path)
    drawplt_line(x=range(1, args.n_epochs+1), y=top1_accs, xlabel='Epoch', ylabel='Validation Top-1 Accuracy', title=f'taskdistill-{args.source_model}-to-{args.target_model}-{args.target_dataset}', savepth=args.save_path)
    drawplt_line(x=range(1, args.n_epochs+1), y=top5_accs, xlabel='Epoch', ylabel='Validation Top-5 Accuracy', title=f'taskdistill-{args.source_model}-to-{args.target_model}-{args.target_dataset}', savepth=args.save_path)
    drawplt_line(x=range(1, args.n_epochs+1), y=epoch_times, xlabel='Epoch', ylabel='Time Cost (seconds)', title=f'taskdistill-{args.source_model}-to-{args.target_model}-{args.target_dataset}', savepth=args.save_path)
                        
    print('Finished taskdistill')
    
    finetune_time = sum(epoch_times)
    return finetune_time