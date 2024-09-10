import torch
import time

def evaluate_model(model, dataloader, criterion,attacker = None):
    start_time = time.time()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")            
    # model.cuda()
    model.to(device)
    model.eval()
    
    top1_correct = 0
    top5_correct = 0
    total = 0

    total_loss = 0.0
    print("len(dataloader.dataset):",len(dataloader.dataset))
    
    
    # for inputs, labels in dataloader:
    for i, (inputs, labels) in enumerate(dataloader, 0):
        # print(f"Inputs type: {type(inputs)}, Labels type: {type(labels)}")
        # Inputs type: <class 'torch.Tensor'>, Labels type: <class 'int'>

        # print("labels:",labels)
        #     # Check if any label is out of bounds
        # assert (labels >= 0).all() and (labels < 10).all()
    
        if len(dataloader) < 30:
            print(f'batch index: {i}')
        # print("inputs.shape:",inputs.shape)
        # print("labels.shape:",labels.shape)
        # print("labels:",labels)
        if attacker == None:
            inputs, labels = inputs.to(device), labels.to(device)
        else:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = attacker(inputs, labels)
        with torch.no_grad():
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
        total_loss += loss.item() * inputs.size(0)
        
        # Top1 accuracy
        _, predicted = torch.max(outputs, 1)
        top1_correct += (predicted == labels).sum().item()
        
        # Top5 accuracy
        _, top5_predicted = torch.topk(outputs, 5, dim=1)
        top5_correct += sum(labels[i] in top5_predicted[i] for i in range(labels.size(0)))
        
        total += labels.size(0)
    
    # Compute averages
    top1_accuracy = top1_correct / total
    top5_accuracy = top5_correct / total
    average_loss = total_loss / total
    
    end_time = time.time()
    evaluate_time = end_time - start_time
    
    return [top1_accuracy, top5_accuracy, average_loss, evaluate_time]
