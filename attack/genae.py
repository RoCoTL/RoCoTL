
import torch
from utils.minmaxvalue import get_dataset_min_max
import advertorch.attacks
from art.estimators.classification import PyTorchClassifier
from art.attacks import evasion
import torch.utils.data as Data
import time
import TransferAttackSurrogates.TransferAttack.CIFAR_Train.utils as utils



def genae(model, testloader, criterion, num_classes, args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")     
    model = model.to(device)
                   
    # optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.01, momentum=0.9)
    # 20240719
    if not args.transattacksurrogate:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    else: 
        optimizer = utils.get_optim(
            args.optim, model, args,
            lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum,sam=args.sam, rho=args.rho, swa=args.swa)
    test_min_val, test_max_val = get_dataset_min_max(testloader)
    print(f"Test set - Min value: {test_min_val}, Max value: {test_max_val}")
    print("test_max_val-test_min_val:",(test_max_val-test_min_val))
    print("args.ae_eps:",args.ae_eps)
    print("args.ae_step:",args.ae_step)
    print("args.ae_eps*(test_max_val-test_min_val):",args.ae_eps*(test_max_val-test_min_val))
    print("args.ae_step*(test_max_val-test_min_val):",args.ae_ite*(test_max_val-test_min_val))
    
    ae_eps = (args.ae_eps*(test_max_val-test_min_val))/255
    ae_step = (args.ae_step*(test_max_val-test_min_val))/255

    print("Type of ae_eps:", type(ae_eps))
    print("Type of ae_step:", type(ae_step))

    ae_eps = float(ae_eps)
    ae_step = float(ae_step)

    print("Type of ae_eps:", type(ae_eps))
    print("Type of ae_step:", type(ae_step))
    
    print("ae_eps:", ae_eps)
    print("ae_step:", ae_step)
    """ 
    surrogate-resnet18-generate fgsm-16-4-20 adversarial-imagenette-examples
    Test set - Min value: -2.1179039478302, Max value: 2.640000104904175
    test_max_val-test_min_val: 4.757904052734375
    args.ae_eps: 16
    args.ae_step: 4
    args.ae_eps*(test_max_val-test_min_val): 76.12646484375
    args.ae_step*(test_max_val-test_min_val): 95.1580810546875
    Type of ae_eps: <class 'float'>
    Type of ae_step: <class 'float'>
    Type of ae_eps: <class 'float'>
    Type of ae_step: <class 'float'>
    ae_eps: 0.29853515625
    ae_step: 0.0746337890625
    """           
    # raise Exception('maggie test')

    print("generate transferable adversarial examples")

    if args.attack_type == 'fgsm':
        advgen_model = advertorch.attacks.GradientSignAttack(predict=model, eps=ae_eps, clip_min=test_min_val, clip_max=test_max_val, targeted=False)  

    elif args.attack_type == 'pgd-linf':
        advgen_model = advertorch.attacks.LinfPGDAttack(predict=model, eps=ae_eps, nb_iter = args.ae_ite, eps_iter = ae_step, clip_min=test_min_val, clip_max=test_max_val, targeted=False)      
    
    elif args.attack_type == 'pgd-l2':
        advgen_model = advertorch.attacks.L2PGDAttack(predict=model, eps=ae_eps, nb_iter = args.ae_ite, eps_iter = ae_step, clip_min=test_min_val, clip_max=test_max_val, targeted=False)      

    elif args.attack_type == 'pgd-l1':
        advgen_model = advertorch.attacks.L1PGDAttack(predict=model, eps=ae_eps, nb_iter = args.ae_ite, eps_iter = ae_step, clip_min=test_min_val, clip_max=test_max_val, targeted=False) 

    elif args.attack_type == 'mifgsm-linf':
        advgen_model = advertorch.attacks.LinfMomentumIterativeAttack(predict=model, eps=ae_eps, nb_iter = args.ae_ite, eps_iter = ae_step, clip_min=test_min_val, clip_max=test_max_val, targeted=False) 

    elif args.attack_type == 'mifgsm-l2':
        advgen_model = advertorch.attacks.L2MomentumIterativeAttack(predict=model, eps=ae_eps, nb_iter = args.ae_ite, eps_iter = ae_step, clip_min=test_min_val, clip_max=test_max_val, targeted=False) 
        
    elif args.attack_type == 'fab-linf':
        advgen_model = advertorch.attacks.LinfFABAttack(predict=model, n_restarts = 1, n_iter = 100, eps=None, alpha_max = 0.1, eta=1.05, beta=0.9) 
        
    elif args.attack_type == 'fab-l2':
        advgen_model = advertorch.attacks.L2FABAttack(predict=model, n_restarts = 1, n_iter = 100, eps=None, alpha_max = 0.1, eta=1.05, beta=0.9) 
        
    elif args.attack_type == 'fab-l1':
        advgen_model = advertorch.attacks.L1FABAttack(predict=model, n_restarts = 1, n_iter = 100, eps=None, alpha_max = 0.1, eta=1.05, beta=0.9) 
        
    elif args.attack_type == 'cw-l2':
        advgen_model = advertorch.attacks.CarliniWagnerL2Attack(predict=model, num_classes = num_classes, confidence = 0, targeted = False, learning_rate=0.01, binary_search_steps=9, max_iterations=10000, abort_early=True, initial_const=1e-3, clip_min=test_min_val, clip_max=test_max_val) 
        
    elif args.attack_type == 'autoattack':

        sample,_ = testloader.dataset[0]
        print("sample.shape:",sample.shape)
        # sample.shape: torch.Size([3, 224, 224])
        print("num_classes:",num_classes)
        
        classifier = PyTorchClassifier(model=model, loss=criterion, input_shape=sample.shape, nb_classes=num_classes, optimizer=optimizer, clip_values=(test_min_val, test_max_val))
        advgen_model = evasion.AutoAttack(estimator=classifier, eps=ae_eps, eps_step = ae_step, batch_size=args.batch_size, targeted = False)
        
    elif args.attack_type == 'deepfool':

        sample,_ = testloader.dataset[0]
        print("sample.shape:",sample.shape)
        # sample.shape: torch.Size([3, 224, 224])
        print("num_classes:",num_classes)
        
        classifier = PyTorchClassifier(model=model, loss=criterion, input_shape=sample.shape, nb_classes=num_classes, optimizer=optimizer, clip_values=(test_min_val, test_max_val))
        advgen_model = evasion.DeepFool(classifier=classifier, max_iter = 100, epsilon=ae_eps, nb_grads = 10, batch_size=args.batch_size)
        
    elif args.attack_type == 'universal':

        sample,_ = testloader.dataset[0]
        print("sample.shape:",sample.shape)
        # sample.shape: torch.Size([3, 224, 224])
        print("num_classes:",num_classes)
        
        classifier = PyTorchClassifier(model=model, loss=criterion, input_shape=sample.shape, nb_classes=num_classes, optimizer=optimizer, clip_values=(test_min_val, test_max_val))
        advgen_model = evasion.UniversalPerturbation(classifier=classifier, attacker = 'pgd', delta = 0.2, max_iter = 100, eps=ae_eps, norm='inf', batch_size=args.batch_size)  
    
    elif args.attack_type == 'zoo':

        sample,_ = testloader.dataset[0]
        print("sample.shape:",sample.shape)
        # sample.shape: torch.Size([3, 224, 224])
        print("num_classes:",num_classes)
        
        classifier = PyTorchClassifier(model=model, loss=criterion, input_shape=sample.shape, nb_classes=num_classes, optimizer=optimizer, clip_values=(test_min_val, test_max_val))
        advgen_model = evasion.ZooAttack(classifier=classifier, confidence=0, targeted=False, max_iter = args.ae_ite, batch_size=args.batch_size) 
        
        
    elif args.attack_type == 'boundary':

        sample,_ = testloader.dataset[0]
        print("sample.shape:",sample.shape)
        # sample.shape: torch.Size([3, 224, 224])
        print("num_classes:",num_classes)
        
        classifier = PyTorchClassifier(model=model, loss=criterion, input_shape=sample.shape, nb_classes=num_classes, optimizer=optimizer, clip_values=(test_min_val, test_max_val))
        advgen_model = evasion.BoundaryAttack(estimator=classifier, batch_size=args.batch_size,targeted=False, epsilon = ae_eps) 
        
    elif args.attack_type == 'hopskipjump':

        sample,_ = testloader.dataset[0]
        print("sample.shape:",sample.shape)
        # sample.shape: torch.Size([3, 224, 224])
        print("num_classes:",num_classes)
        
        classifier = PyTorchClassifier(model=model, loss=criterion, input_shape=sample.shape, nb_classes=num_classes, optimizer=optimizer, clip_values=(test_min_val, test_max_val))
        advgen_model = evasion.HopSkipJump(classifier=classifier, batch_size=args.batch_size, targeted=False, norm="inf")        
        
    elif args.attack_type == 'simba':

        sample,_ = testloader.dataset[0]
        print("sample.shape:",sample.shape)
        # sample.shape: torch.Size([3, 224, 224])
        print("num_classes:",num_classes)
        
        classifier = PyTorchClassifier(model=model, loss=criterion, input_shape=sample.shape, nb_classes=num_classes, optimizer=optimizer, clip_values=(test_min_val, test_max_val))
        advgen_model = evasion.SimBA(classifier=classifier, epsilon=ae_eps, targeted=False, batch_size=1)       
        """ 
        ValueError: The batch size `batch_size` has to be 1 in this implementation.
        """
        
    elif args.attack_type == 'square':

        sample,_ = testloader.dataset[0]
        print("sample.shape:",sample.shape)
        # sample.shape: torch.Size([3, 224, 224])
        print("num_classes:",num_classes)
        
        classifier = PyTorchClassifier(model=model, loss=criterion, input_shape=sample.shape, nb_classes=num_classes, optimizer=optimizer, clip_values=(test_min_val, test_max_val))
        advgen_model = evasion.SquareAttack(estimator=classifier, norm='inf',eps=ae_eps, batch_size=args.batch_size)               
        
        
        
    print("BacthNum = len(testloader):",len(testloader))
    print("SampleNum = len(testloader.dataset):",len(testloader.dataset))
              
    start_time = time.time()   
    for batch_index, (inputs, labels) in enumerate(testloader):
        print("batch_index:",batch_index)
        inputs = inputs.to(device)
        labels = labels.to(device)

        if args.attack_type in ['fgsm', 'pgd-linf', 'pgd-l2', 'pgd-l1', 'mifgsm-linf', 'mifgsm-l2', 'cw-l2']: 
            adv_batch_x = advgen_model.perturb(x=inputs, y=labels)

        if args.attack_type in ['fab-linf', 'fab-l2','fab-l1']: 
            adv_batch_x = advgen_model.perturb(x=inputs, y=labels)
        
        if args.attack_type in ['autoattack','deepfool','zoo','boundary','hopskipjump','simba','square']: 
            adv_batch_x = advgen_model.generate(x=inputs.cpu().numpy())
            adv_batch_x = torch.from_numpy(adv_batch_x).to(device)

        if args.attack_type in ['universal']: 
            adv_batch_x = advgen_model.generate(x=inputs.cpu().numpy())
            adv_batch_x = torch.from_numpy(adv_batch_x).to(device)
                    
        adv_batch_y = labels        

        if batch_index == 0:
            adv_testset_x = adv_batch_x
            adv_testset_y = adv_batch_y
        else:
            adv_testset_x = torch.cat((adv_testset_x, adv_batch_x), dim=0)
            adv_testset_y = torch.cat((adv_testset_y, adv_batch_y), dim=0)


        end_time = time.time()   
        genrate_time = end_time - start_time   



    if args.surrogate_dataset == 'imagenette':
        classes = [('tench', 'Tinca tinca'), ('English springer', 'English springer spaniel'), ('cassette player',), ('chain saw', 'chainsaw'), ('church', 'church building'), ('French horn', 'horn'), ('garbage truck', 'dustcart'), ('gas pump', 'gasoline pump', 'petrol pump', 'island dispenser'), ('golf ball',), ('parachute', 'chute')]
        
        class_to_idx = {'tench': 0, 'Tinca tinca': 0, 'English springer': 1, 'English springer spaniel': 1, 'cassette player': 2, 'chain saw': 3, 'chainsaw': 3, 'church': 4, 'church building': 4, 'French horn': 5, 'horn': 5, 'garbage truck': 6, 'dustcart': 6, 'gas pump': 7, 'gasoline pump': 7, 'petrol pump': 7, 'island dispenser': 7, 'golf ball': 8, 'parachute': 9, 'chute': 9}        


    elif args.surrogate_dataset == 'imagewoof':
        classes = ['n02086240', 'n02087394', 'n02088364', 'n02089973', 'n02093754', 'n02096294', 'n02099601', 'n02105641', 'n02111889', 'n02115641']
        
        class_to_idx = {'n02086240': 0, 'n02087394': 1, 'n02088364': 2, 'n02089973': 3, 'n02093754': 4, 'n02096294': 5, 'n02099601': 6, 'n02105641': 7, 'n02111889': 8, 'n02115641': 9}
        

    elif args.surrogate_dataset == 'tiny_imagenet':
        # classes = [f'n{i:08}' for i in range(200)]  # TinyImageNet 的类标签以 'n' 开头并有八个字符

        classes = ['n01443537', 'n01629819', 'n01641577', 'n01644900', 'n01698640', 'n01742172', 'n01768244', 'n01770393', 'n01774384', 'n01774750', 'n01784675', 'n01855672', 'n01882714', 'n01910747', 'n01917289', 'n01944390', 'n01945685', 'n01950731', 'n01983481', 'n01984695', 'n02002724', 'n02056570', 'n02058221', 'n02074367', 'n02085620', 'n02094433', 'n02099601', 'n02099712', 'n02106662', 'n02113799', 'n02123045', 'n02123394', 'n02124075', 'n02125311', 'n02129165', 'n02132136', 'n02165456', 'n02190166', 'n02206856', 'n02226429', 'n02231487', 'n02233338', 'n02236044', 'n02268443', 'n02279972', 'n02281406', 'n02321529', 'n02364673', 'n02395406', 'n02403003', 'n02410509', 'n02415577', 'n02423022', 'n02437312', 'n02480495', 'n02481823', 'n02486410', 'n02504458', 'n02509815', 'n02666196', 'n02669723', 'n02699494', 'n02730930', 'n02769748', 'n02788148', 'n02791270', 'n02793495', 'n02795169', 'n02802426', 'n02808440', 'n02814533', 'n02814860', 'n02815834', 'n02823428', 'n02837789', 'n02841315', 'n02843684', 'n02883205', 'n02892201', 'n02906734', 'n02909870', 'n02917067', 'n02927161', 'n02948072', 'n02950826', 'n02963159', 'n02977058', 'n02988304', 'n02999410', 'n03014705', 'n03026506', 'n03042490', 'n03085013', 'n03089624', 'n03100240', 'n03126707', 'n03160309', 'n03179701', 'n03201208', 'n03250847', 'n03255030', 'n03355925', 'n03388043', 'n03393912', 'n03400231', 'n03404251', 'n03424325', 'n03444034', 'n03447447', 'n03544143', 'n03584254', 'n03599486', 'n03617480', 'n03637318', 'n03649909', 'n03662601', 'n03670208', 'n03706229', 'n03733131', 'n03763968', 'n03770439', 'n03796401', 'n03804744', 'n03814639', 'n03837869', 'n03838899', 'n03854065', 'n03891332', 'n03902125', 'n03930313', 'n03937543', 'n03970156', 'n03976657', 'n03977966', 'n03980874', 'n03983396', 'n03992509', 'n04008634', 'n04023962', 'n04067472', 'n04070727', 'n04074963', 'n04099969', 'n04118538', 'n04133789', 'n04146614', 'n04149813', 'n04179913', 'n04251144', 'n04254777', 'n04259630', 'n04265275', 'n04275548', 'n04285008', 'n04311004', 'n04328186', 'n04356056', 'n04366367', 'n04371430', 'n04376876', 'n04398044', 'n04399382', 'n04417672', 'n04456115', 'n04465501', 'n04486054', 'n04487081', 'n04501370', 'n04507155', 'n04532106', 'n04532670', 'n04540053', 'n04560804', 'n04562935', 'n04596742', 'n04597913', 'n06596364', 'n07579787', 'n07583066', 'n07614500', 'n07615774', 'n07695742', 'n07711569', 'n07715103', 'n07720875', 'n07734744', 'n07747607', 'n07749582', 'n07753592', 'n07768694', 'n07871810', 'n07873807', 'n07875152', 'n07920052', 'n09193705', 'n09246464', 'n09256479', 'n09332890', 'n09428293', 'n12267677']
        
        class_to_idx = {'n01443537': 0, 'n01629819': 1, 'n01641577': 2, 'n01644900': 3, 'n01698640': 4, 'n01742172': 5, 'n01768244': 6, 'n01770393': 7, 'n01774384': 8, 'n01774750': 9, 'n01784675': 10, 'n01855672': 11, 'n01882714': 12, 'n01910747': 13, 'n01917289': 14, 'n01944390': 15, 'n01945685': 16, 'n01950731': 17, 'n01983481': 18, 'n01984695': 19, 'n02002724': 20, 'n02056570': 21, 'n02058221': 22, 'n02074367': 23, 'n02085620': 24, 'n02094433': 25, 'n02099601': 26, 'n02099712': 27, 'n02106662': 28, 'n02113799': 29, 'n02123045': 30, 'n02123394': 31, 'n02124075': 32, 'n02125311': 33, 'n02129165': 34, 'n02132136': 35, 'n02165456': 36, 'n02190166': 37, 'n02206856': 38, 'n02226429': 39, 'n02231487': 40, 'n02233338': 41, 'n02236044': 42, 'n02268443': 43, 'n02279972': 44, 'n02281406': 45, 'n02321529': 46, 'n02364673': 47, 'n02395406': 48, 'n02403003': 49, 'n02410509': 50, 'n02415577': 51, 'n02423022': 52, 'n02437312': 53, 'n02480495': 54, 'n02481823': 55, 'n02486410': 56, 'n02504458': 57, 'n02509815': 58, 'n02666196': 59, 'n02669723': 60, 'n02699494': 61, 'n02730930': 62, 'n02769748': 63, 'n02788148': 64, 'n02791270': 65, 'n02793495': 66, 'n02795169': 67, 'n02802426': 68, 'n02808440': 69, 'n02814533': 70, 'n02814860': 71, 'n02815834': 72, 'n02823428': 73, 'n02837789': 74, 'n02841315': 75, 'n02843684': 76, 'n02883205': 77, 'n02892201': 78, 'n02906734': 79, 'n02909870': 80, 'n02917067': 81, 'n02927161': 82, 'n02948072': 83, 'n02950826': 84, 'n02963159': 85, 'n02977058': 86, 'n02988304': 87, 'n02999410': 88, 'n03014705': 89, 'n03026506': 90, 'n03042490': 91, 'n03085013': 92, 'n03089624': 93, 'n03100240': 94, 'n03126707': 95, 'n03160309': 96, 'n03179701': 97, 'n03201208': 98, 'n03250847': 99, 'n03255030': 100, 'n03355925': 101, 'n03388043': 102, 'n03393912': 103, 'n03400231': 104, 'n03404251': 105, 'n03424325': 106, 'n03444034': 107, 'n03447447': 108, 'n03544143': 109, 'n03584254': 110, 'n03599486': 111, 'n03617480': 112, 'n03637318': 113, 'n03649909': 114, 'n03662601': 115, 'n03670208': 116, 'n03706229': 117, 'n03733131': 118, 'n03763968': 119, 'n03770439': 120, 'n03796401': 121, 'n03804744': 122, 'n03814639': 123, 'n03837869': 124, 'n03838899': 125, 'n03854065': 126, 'n03891332': 127, 'n03902125': 128, 'n03930313': 129, 'n03937543': 130, 'n03970156': 131, 'n03976657': 132, 'n03977966': 133, 'n03980874': 134, 'n03983396': 135, 'n03992509': 136, 'n04008634': 137, 'n04023962': 138, 'n04067472': 139, 'n04070727': 140, 'n04074963': 141, 'n04099969': 142, 'n04118538': 143, 'n04133789': 144, 'n04146614': 145, 'n04149813': 146, 'n04179913': 147, 'n04251144': 148, 'n04254777': 149, 'n04259630': 150, 'n04265275': 151, 'n04275548': 152, 'n04285008': 153, 'n04311004': 154, 'n04328186': 155, 'n04356056': 156, 'n04366367': 157, 'n04371430': 158, 'n04376876': 159, 'n04398044': 160, 'n04399382': 161, 'n04417672': 162, 'n04456115': 163, 'n04465501': 164, 'n04486054': 165, 'n04487081': 166, 'n04501370': 167, 'n04507155': 168, 'n04532106': 169, 'n04532670': 170, 'n04540053': 171, 'n04560804': 172, 'n04562935': 173, 'n04596742': 174, 'n04597913': 175, 'n06596364': 176, 'n07579787': 177, 'n07583066': 178, 'n07614500': 179, 'n07615774': 180, 'n07695742': 181, 'n07711569': 182, 'n07715103': 183, 'n07720875': 184, 'n07734744': 185, 'n07747607': 186, 'n07749582': 187, 'n07753592': 188, 'n07768694': 189, 'n07871810': 190, 'n07873807': 191, 'n07875152': 192, 'n07920052': 193, 'n09193705': 194, 'n09246464': 195, 'n09256479': 196, 'n09332890': 197, 'n09428293': 198, 'n12267677': 199}
        
    else:
        raise ValueError("Unsupported dataset. Choose either 'imagenette' or 'tiny_imagenet'.")

    # 创建 TensorDataset
    adv_testset = Data.TensorDataset(adv_testset_x, adv_testset_y)
    adv_testset.data = adv_testset.tensors[0]
    adv_testset.targets = adv_testset.tensors[1]
    adv_testset.classes = classes
    # adv_testset.classes_to_idx = {label: i for i, label in enumerate(classes)}
    adv_testset.class_to_idx = class_to_idx

    adv_testloader = Data.DataLoader(adv_testset, args.batch_size, shuffle=False)

    print("BacthNum = len(adv_testloader):",len(adv_testloader))
    print("SampleNum = len(adv_testloader.dataset):",len(adv_testloader.dataset))
    
    return adv_testloader, genrate_time


