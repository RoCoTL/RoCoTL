import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data import Subset
import os
from PIL import Image
from utils.mapclassidx import *
import copy
import torch.utils.data
import json
from dataset.datasetcard import *

base_path = 'dataset_path'  # if use alchem server, use this

class AlignedImageNetteDataset(torch.utils.data.Dataset):
    def __init__(self, root, split='val', transform=None, class_to_idx=None):
        self.dataset = datasets.Imagenette(root=root, split=split, transform=transform)
        self.class_to_idx = class_to_idx
        self.classes = self.dataset.classes
        
        desired_class_indices = [self.dataset.class_to_idx[class_name] for class_name in self.class_to_idx.keys()]


        filtered_indices = []

        for idx, (_, target) in enumerate(self.dataset):
            if target in desired_class_indices:
                filtered_indices.append(idx)


        self.dataset = Subset(self.dataset, filtered_indices)
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        class_names = self.dataset.dataset.classes[label]
        class_name = class_names[0]  # Assuming the first name is the primary class identifier
        label_idx = self.class_to_idx[class_name]
        return img, label_idx
    
class AlignedImageWoofDataset(torch.utils.data.Dataset):
    def __init__(self, root, split='val', transform=None, class_to_idx=None):
        # self.dataset = datasets.Imagenette(root=root, split=split, transform=transform)
        self.dataset = datasets.ImageFolder(root=root, transform=transform)  
        self.class_to_idx = class_to_idx
        self.classes = self.dataset.classes
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        class_name = self.dataset.classes[label]
        # print("class_name:",class_name)
        # class_names: n02086240
        # class_name = class_names[0]  # Assuming the first name is the primary class identifier
        # label_idx = self.class_to_idx[class_name]
        label_idx = self.class_to_idx[class_name]
        return img, label_idx
    
class AlignedTinyImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, root, split='val', transform=None, class_to_idx=None):
        trainset = datasets.ImageFolder(root=base_path +'/data/TinyImageNet/tiny-imagenet-200/train', transform=transform)  
        self.dataset = TinyImageNetValDataset(root=root, class_to_idx=trainset.class_to_idx, transform=transform)
        self.class_to_idx = class_to_idx
        # print("self.class_to_idx:", self.class_to_idx)
        self.classes = self.dataset.classes
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        class_name = self.dataset.classes[label]
        # print("class_name:",class_name)
        # class_name: n04008634
        # class_name = class_names[0]  # Assuming the first name is the primary class identifier
        # label_idx = self.class_to_idx[class_name]
        label_idx = self.class_to_idx[class_name]
        return img, label_idx
    
class TinyImageNetValDataset(Dataset):
    def __init__(self, root, class_to_idx, transform=None):
        self.root = root
        self.transform = transform
        self.img_dir = os.path.join(root, 'images')
        self.annotations_file = os.path.join(root, 'val_annotations.txt')
        self.class_to_idx = class_to_idx
        self.img_labels = self._load_annotations()
        self.classes = list(class_to_idx.keys())  # Add classes attribute


    def _load_annotations(self):
        img_labels = {}
        with open(self.annotations_file, 'r') as file:
            for line in file:
                parts = line.strip().split('\t')
                img_labels[parts[0]] = self.class_to_idx[parts[1]]
        return img_labels

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = list(self.img_labels.keys())[idx]
        label = self.img_labels[img_name]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class FilteredDataset(Dataset):
    def __init__(self, dataset, allowed_labels,name):
        self.dataset = dataset
        self.allowed_labels = set(allowed_labels)
        if name == 'cifar10':
            mask = torch.isin(torch.tensor(self.dataset.targets), torch.tensor(list(self.allowed_labels)))
        elif name == 'cifar100':
            mask = torch.isin(torch.tensor(self.dataset.targets), torch.tensor(list(self.allowed_labels)))
        self.filtered_indices = torch.nonzero(mask).flatten().tolist()
        # self.filtered_indices = [i for i, (_, label) in enumerate(dataset) if label in self.allowed_labels]
        print("self.filtered_indices:",self.filtered_indices)
        torch.save(self.filtered_indices, 'dataset/filtered_tinyimagenet_indices.pth')
        
    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        return self.dataset[self.filtered_indices[idx]]

#============================================================================================

def get_source_dataset(args):
    
    print(f'args.source_dataset is {args.source_dataset}')
    
    if args.source_dataset == 'imagenet':

        transform_test = transforms.Compose([
            transforms.Resize(size=224),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        source_test_dataset = datasets.ImageNet(root= base_path +'/data/ImageNet2012', split='val', transform=transform_test)
        print(f'Test dataset size: {len(source_test_dataset)}')
        
        source_trainset = datasets.ImageNet(root=base_path + '/data/ImageNet2012', split='train', transform=transform_test)
        print(f'Original training dataset size: {len(source_trainset)}')
        
        train_size = int(0.8 * len(source_trainset))
        val_size = len(source_trainset) - train_size

        source_trainset, source_valset = torch.utils.data.random_split(source_trainset, [train_size, val_size])
        
        print(f'Splitted training dataset size: {len(source_trainset)}')
        print(f'Splitted validation dataset size: {len(source_valset)}')
        """ 
        Test dataset size:                 50,000
        Original training dataset size: 1,281,167
        Splitted training dataset size: 1,024,933
        Splitted validation dataset size: 256,234
        len(dataloader.dataset): 50000
        """
            
    else:
        raise Exception('please input the valid source dataset') 
    
    return source_trainset, source_valset, source_test_dataset

def get_surrogate_dataset(args, clean=False, adv=False):
    
    print(f'args.surrogate_dataset is {args.surrogate_dataset}')
    if clean == True:

        if args.surrogate_dataset == 'imagenette':
            
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(size=224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            transform_test = transforms.Compose([
                transforms.Resize(size=224),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            surrogate_testset = datasets.Imagenette(root=base_path +'/data/ImageNette', split='val', transform=transform_test)
            print(f'Test dataset size: {len(surrogate_testset)}')

            surrogate_trainset = datasets.Imagenette(root= base_path +'/data/ImageNette', split='train', transform=transform_train)
            print(f'Original training dataset size: {len(surrogate_trainset)}')
                
            train_size = int(0.8 * len(surrogate_trainset))
            val_size = len(surrogate_trainset) - train_size

            surrogate_trainset, surrogate_valset = torch.utils.data.random_split(surrogate_trainset, [train_size, val_size])
            
            print(f'Splitted training dataset size: {len(surrogate_trainset)}')
            print(f'Splitted validation dataset size: {len(surrogate_valset)}')

            """ 
            Test dataset size:                  3,925
            Original training dataset size:     9,469
            Splitted training dataset size:     7,575
            Splitted validation dataset size:   1,894
            
            classes: [('tench', 'Tinca tinca'), ('English springer', 'English springer spaniel'), ('cassette player',), ('chain saw', 'chainsaw'), ('church', 'church building'), ('French horn', 'horn'), ('garbage truck', 'dustcart'), ('gas pump', 'gasoline pump', 'petrol pump', 'island dispenser'), ('golf ball',), ('parachute', 'chute')]
            
            class_to_idx: {'tench': 0, 'Tinca tinca': 0, 'English springer': 1, 'English springer spaniel': 1, 'cassette player': 2, 'chain saw': 3, 'chainsaw': 3, 'church': 4, 'church building': 4, 'French horn': 5, 'horn': 5, 'garbage truck': 6, 'dustcart': 6, 'gas pump': 7, 'gasoline pump': 7, 'petrol pump': 7, 'island dispenser': 7, 'golf ball': 8, 'parachute': 9, 'chute': 9}        
            """
            
        elif args.surrogate_dataset == 'imagewoof':  
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(size=224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            transform_test = transforms.Compose([
                transforms.Resize(size=224),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            surrogate_testset = datasets.ImageFolder(root=base_path +'/data/ImageWoof/imagewoof2/val', transform=transform_test)
            print(f'Test dataset size: {len(surrogate_testset)}') 
            
            surrogate_trainset = datasets.ImageFolder(root=base_path +'/data/ImageWoof/imagewoof2/train', transform=transform_train)        
            print(f'Original training dataset size: {len(surrogate_trainset)}')
            # classes = surrogate_trainset.classes
            # print("classes:",classes)
            # class_to_idx = surrogate_trainset.class_to_idx
            # print("class_to_idx:",class_to_idx)
                    
            train_size = int(0.8 * len(surrogate_trainset))
            val_size = len(surrogate_trainset) - train_size

            surrogate_trainset, surrogate_valset = torch.utils.data.random_split(surrogate_trainset, [train_size, val_size])
            
            print(f'Splitted training dataset size: {len(surrogate_trainset)}')
            print(f'Splitted validation dataset size: {len(surrogate_valset)}')

            """ 
            Test dataset size:                  3,929
            Original training dataset size:     9,025
            Splitted training dataset size:     7,220
            Splitted validation dataset size:   1,805
            
            classes: ['n02086240', 'n02087394', 'n02088364', 'n02089973', 'n02093754', 'n02096294', 'n02099601', 'n02105641', 'n02111889', 'n02115641']
            
            class_to_idx: {'n02086240': 0, 'n02087394': 1, 'n02088364': 2, 'n02089973': 3, 'n02093754': 4, 'n02096294': 5, 'n02099601': 6, 'n02105641': 7, 'n02111889': 8, 'n02115641': 9}
            """                
            
        elif args.surrogate_dataset == 'tiny_imagenet':
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(size=224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            transform_test = transforms.Compose([
                transforms.Resize(size=224),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            surrogate_trainset = datasets.ImageFolder(root= base_path +'/data/TinyImageNet/tiny-imagenet-200/train', transform=transform_train)
            print(f'Original training dataset size: {len(surrogate_trainset)}')        
            class_to_idx = surrogate_trainset.class_to_idx

            surrogate_testset = TinyImageNetValDataset(root= base_path +'/data/TinyImageNet/tiny-imagenet-200/val', class_to_idx=class_to_idx, transform=transform_test)
            print(f'Test dataset size: {len(surrogate_testset)}')

            train_size = int(0.8 * len(surrogate_trainset))
            val_size = len(surrogate_trainset) - train_size

            surrogate_trainset, surrogate_valset = torch.utils.data.random_split(surrogate_trainset, [train_size, val_size])
            
            print(f'Splitted training dataset size: {len(surrogate_trainset)}')
            print(f'Splitted validation dataset size: {len(surrogate_valset)}')
            """ 
            Test dataset size:                  10,000
            Original training dataset size:    100,000
            Splitted training dataset size:     80,000
            Splitted validation dataset size:   20,000

            classes: ['n01443537', 'n01629819', 'n01641577', 'n01644900', 'n01698640', 'n01742172', 'n01768244', 'n01770393', 'n01774384', 'n01774750', 'n01784675', 'n01855672', 'n01882714', 'n01910747', 'n01917289', 'n01944390', 'n01945685', 'n01950731', 'n01983481', 'n01984695', 'n02002724', 'n02056570', 'n02058221', 'n02074367', 'n02085620', 'n02094433', 'n02099601', 'n02099712', 'n02106662', 'n02113799', 'n02123045', 'n02123394', 'n02124075', 'n02125311', 'n02129165', 'n02132136', 'n02165456', 'n02190166', 'n02206856', 'n02226429', 'n02231487', 'n02233338', 'n02236044', 'n02268443', 'n02279972', 'n02281406', 'n02321529', 'n02364673', 'n02395406', 'n02403003', 'n02410509', 'n02415577', 'n02423022', 'n02437312', 'n02480495', 'n02481823', 'n02486410', 'n02504458', 'n02509815', 'n02666196', 'n02669723', 'n02699494', 'n02730930', 'n02769748', 'n02788148', 'n02791270', 'n02793495', 'n02795169', 'n02802426', 'n02808440', 'n02814533', 'n02814860', 'n02815834', 'n02823428', 'n02837789', 'n02841315', 'n02843684', 'n02883205', 'n02892201', 'n02906734', 'n02909870', 'n02917067', 'n02927161', 'n02948072', 'n02950826', 'n02963159', 'n02977058', 'n02988304', 'n02999410', 'n03014705', 'n03026506', 'n03042490', 'n03085013', 'n03089624', 'n03100240', 'n03126707', 'n03160309', 'n03179701', 'n03201208', 'n03250847', 'n03255030', 'n03355925', 'n03388043', 'n03393912', 'n03400231', 'n03404251', 'n03424325', 'n03444034', 'n03447447', 'n03544143', 'n03584254', 'n03599486', 'n03617480', 'n03637318', 'n03649909', 'n03662601', 'n03670208', 'n03706229', 'n03733131', 'n03763968', 'n03770439', 'n03796401', 'n03804744', 'n03814639', 'n03837869', 'n03838899', 'n03854065', 'n03891332', 'n03902125', 'n03930313', 'n03937543', 'n03970156', 'n03976657', 'n03977966', 'n03980874', 'n03983396', 'n03992509', 'n04008634', 'n04023962', 'n04067472', 'n04070727', 'n04074963', 'n04099969', 'n04118538', 'n04133789', 'n04146614', 'n04149813', 'n04179913', 'n04251144', 'n04254777', 'n04259630', 'n04265275', 'n04275548', 'n04285008', 'n04311004', 'n04328186', 'n04356056', 'n04366367', 'n04371430', 'n04376876', 'n04398044', 'n04399382', 'n04417672', 'n04456115', 'n04465501', 'n04486054', 'n04487081', 'n04501370', 'n04507155', 'n04532106', 'n04532670', 'n04540053', 'n04560804', 'n04562935', 'n04596742', 'n04597913', 'n06596364', 'n07579787', 'n07583066', 'n07614500', 'n07615774', 'n07695742', 'n07711569', 'n07715103', 'n07720875', 'n07734744', 'n07747607', 'n07749582', 'n07753592', 'n07768694', 'n07871810', 'n07873807', 'n07875152', 'n07920052', 'n09193705', 'n09246464', 'n09256479', 'n09332890', 'n09428293', 'n12267677']
            
            class_to_idx: {'n01443537': 0, 'n01629819': 1, 'n01641577': 2, 'n01644900': 3, 'n01698640': 4, 'n01742172': 5, 'n01768244': 6, 'n01770393': 7, 'n01774384': 8, 'n01774750': 9, 'n01784675': 10, 'n01855672': 11, 'n01882714': 12, 'n01910747': 13, 'n01917289': 14, 'n01944390': 15, 'n01945685': 16, 'n01950731': 17, 'n01983481': 18, 'n01984695': 19, 'n02002724': 20, 'n02056570': 21, 'n02058221': 22, 'n02074367': 23, 'n02085620': 24, 'n02094433': 25, 'n02099601': 26, 'n02099712': 27, 'n02106662': 28, 'n02113799': 29, 'n02123045': 30, 'n02123394': 31, 'n02124075': 32, 'n02125311': 33, 'n02129165': 34, 'n02132136': 35, 'n02165456': 36, 'n02190166': 37, 'n02206856': 38, 'n02226429': 39, 'n02231487': 40, 'n02233338': 41, 'n02236044': 42, 'n02268443': 43, 'n02279972': 44, 'n02281406': 45, 'n02321529': 46, 'n02364673': 47, 'n02395406': 48, 'n02403003': 49, 'n02410509': 50, 'n02415577': 51, 'n02423022': 52, 'n02437312': 53, 'n02480495': 54, 'n02481823': 55, 'n02486410': 56, 'n02504458': 57, 'n02509815': 58, 'n02666196': 59, 'n02669723': 60, 'n02699494': 61, 'n02730930': 62, 'n02769748': 63, 'n02788148': 64, 'n02791270': 65, 'n02793495': 66, 'n02795169': 67, 'n02802426': 68, 'n02808440': 69, 'n02814533': 70, 'n02814860': 71, 'n02815834': 72, 'n02823428': 73, 'n02837789': 74, 'n02841315': 75, 'n02843684': 76, 'n02883205': 77, 'n02892201': 78, 'n02906734': 79, 'n02909870': 80, 'n02917067': 81, 'n02927161': 82, 'n02948072': 83, 'n02950826': 84, 'n02963159': 85, 'n02977058': 86, 'n02988304': 87, 'n02999410': 88, 'n03014705': 89, 'n03026506': 90, 'n03042490': 91, 'n03085013': 92, 'n03089624': 93, 'n03100240': 94, 'n03126707': 95, 'n03160309': 96, 'n03179701': 97, 'n03201208': 98, 'n03250847': 99, 'n03255030': 100, 'n03355925': 101, 'n03388043': 102, 'n03393912': 103, 'n03400231': 104, 'n03404251': 105, 'n03424325': 106, 'n03444034': 107, 'n03447447': 108, 'n03544143': 109, 'n03584254': 110, 'n03599486': 111, 'n03617480': 112, 'n03637318': 113, 'n03649909': 114, 'n03662601': 115, 'n03670208': 116, 'n03706229': 117, 'n03733131': 118, 'n03763968': 119, 'n03770439': 120, 'n03796401': 121, 'n03804744': 122, 'n03814639': 123, 'n03837869': 124, 'n03838899': 125, 'n03854065': 126, 'n03891332': 127, 'n03902125': 128, 'n03930313': 129, 'n03937543': 130, 'n03970156': 131, 'n03976657': 132, 'n03977966': 133, 'n03980874': 134, 'n03983396': 135, 'n03992509': 136, 'n04008634': 137, 'n04023962': 138, 'n04067472': 139, 'n04070727': 140, 'n04074963': 141, 'n04099969': 142, 'n04118538': 143, 'n04133789': 144, 'n04146614': 145, 'n04149813': 146, 'n04179913': 147, 'n04251144': 148, 'n04254777': 149, 'n04259630': 150, 'n04265275': 151, 'n04275548': 152, 'n04285008': 153, 'n04311004': 154, 'n04328186': 155, 'n04356056': 156, 'n04366367': 157, 'n04371430': 158, 'n04376876': 159, 'n04398044': 160, 'n04399382': 161, 'n04417672': 162, 'n04456115': 163, 'n04465501': 164, 'n04486054': 165, 'n04487081': 166, 'n04501370': 167, 'n04507155': 168, 'n04532106': 169, 'n04532670': 170, 'n04540053': 171, 'n04560804': 172, 'n04562935': 173, 'n04596742': 174, 'n04597913': 175, 'n06596364': 176, 'n07579787': 177, 'n07583066': 178, 'n07614500': 179, 'n07615774': 180, 'n07695742': 181, 'n07711569': 182, 'n07715103': 183, 'n07720875': 184, 'n07734744': 185, 'n07747607': 186, 'n07749582': 187, 'n07753592': 188, 'n07768694': 189, 'n07871810': 190, 'n07873807': 191, 'n07875152': 192, 'n07920052': 193, 'n09193705': 194, 'n09246464': 195, 'n09256479': 196, 'n09332890': 197, 'n09428293': 198, 'n12267677': 199}

            """                                 
        
        else:
            raise Exception('please input the valid surrogate dataset') 
    
    elif adv == True:
        print("args.surrogate_advdataset_x_path:",args.surrogate_advdataset_x_path)
        print("args.surrogate_advdataset_y_path:",args.surrogate_advdataset_y_path)            
        adv_images = torch.load(args.surrogate_advdataset_x_path)  
        adv_labels = torch.load(args.surrogate_advdataset_y_path)  
        print("adv_labels:",adv_labels)
        surrogate_testset = TensorDataset(adv_images, adv_labels)
        surrogate_trainset = None
        surrogate_valset = None
                            
    return surrogate_trainset, surrogate_valset, surrogate_testset

def get_align_surrogate_dataset(args, clean=False, adv=False):
    
    print(f'args.surrogate_dataset is {args.surrogate_dataset}')

    if clean == True:
        if args.taskmode == 'transfer_attack':
            if args.surrogate_dataset == 'imagenette':
                
                transform_train = transforms.Compose([
                    transforms.RandomResizedCrop(size=224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

                transform_test = transforms.Compose([
                    transforms.Resize(size=224),
                    transforms.CenterCrop(size=224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

                # test set
                surrogate_testset = datasets.Imagenette(root=base_path +'/data/ImageNette', split='val', transform=transform_test)
                print(f'Test dataset size: {len(surrogate_testset)}')
                # print("Original surrogate_testset.classes:", surrogate_testset.classes)
                # print("Original surrogate_testset.class_to_idx:", surrogate_testset.class_to_idx)
                imagenet_testset = datasets.ImageNet(root=base_path +'/data/ImageNet2012', split='val')       
                if args.all:
                    updated_classtoidx_surrogate_testset = AlignClassToIdx_ImageNette(surrogate_testset, imagenet_testset.class_to_idx)        
                else:
                    updated_classtoidx_surrogate_testset = AlignClassToIdx_ImageNette2(surrogate_testset, imagenet_testset.class_to_idx,args.target_dataset)      
                # print("updated_classtoidx_surrogate_testset.classes:", updated_classtoidx_surrogate_testset.classes)
                print("updated_classtoidx_surrogate_testset.class_to_idx:", updated_classtoidx_surrogate_testset.class_to_idx)
                align_surrogate_testset = AlignedImageNetteDataset(root=base_path +'/data/ImageNette', split='val', transform=transform_test, class_to_idx=updated_classtoidx_surrogate_testset.class_to_idx)
                # print("align_surrogate_testset.classes:", align_surrogate_testset.classes)
                # print("align_surrogate_testset.class_to_idx:", align_surrogate_testset.class_to_idx)
                print(f'align_surrogate_testset size: {len(align_surrogate_testset.dataset)}')
            
                """ 
                args.surrogate_dataset is imagenette
                Test dataset size: 3925
                
                Original surrogate_testset.classes: 
                [('tench', 'Tinca tinca'), ('English springer', 'English springer spaniel'), ('cassette player',), ('chain saw', 'chainsaw'), ('church', 'church building'), ('French horn', 'horn'), ('garbage truck', 'dustcart'), ('gas pump', 'gasoline pump', 'petrol pump', 'island dispenser'), ('golf ball',), ('parachute', 'chute')]
                
                Original surrogate_testset.class_to_idx: 
                {'tench': 0, 'Tinca tinca': 0, 'English springer': 1, 'English springer spaniel': 1, 'cassette player': 2, 'chain saw': 3, 'chainsaw': 3, 'church': 4, 'church building': 4, 'French horn': 5, 'horn': 5, 'garbage truck': 6, 'dustcart': 6, 'gas pump': 7, 'gasoline pump': 7, 'petrol pump': 7, 'island dispenser': 7, 'golf ball': 8, 'parachute': 9, 'chute': 9}
                
                updated_classtoidx_surrogate_testset.classes: 
                [('tench', 'Tinca tinca'), ('English springer', 'English springer spaniel'), ('cassette player',), ('chain saw', 'chainsaw'), ('church', 'church building'), ('French horn', 'horn'), ('garbage truck', 'dustcart'), ('gas pump', 'gasoline pump', 'petrol pump', 'island dispenser'), ('golf ball',), ('parachute', 'chute')]
                
                updated_classtoidx_surrogate_testset.class_to_idx: 
                {'tench': 0, 'Tinca tinca': 0, 'English springer': 217, 'English springer spaniel': 217, 'cassette player': 482, 'chain saw': 491, 'chainsaw': 491, 'church': 497, 'church building': 497, 'French horn': 566, 'horn': 566, 'garbage truck': 569, 'dustcart': 569, 'gas pump': 571, 'gasoline pump': 571, 'petrol pump': 571, 'island dispenser': 571, 'golf ball': 574, 'parachute': 701, 'chute': 701}
                
                align_surrogate_testset.classes: 
                [('tench', 'Tinca tinca'), ('English springer', 'English springer spaniel'), ('cassette player',), ('chain saw', 'chainsaw'), ('church', 'church building'), ('French horn', 'horn'), ('garbage truck', 'dustcart'), ('gas pump', 'gasoline pump', 'petrol pump', 'island dispenser'), ('golf ball',), ('parachute', 'chute')]
                
                align_surrogate_testset.class_to_idx: 
                {'tench': 0, 'Tinca tinca': 0, 'English springer': 217, 'English springer spaniel': 217, 'cassette player': 482, 'chain saw': 491, 'chainsaw': 491, 'church': 497, 'church building': 497, 'French horn': 566, 'horn': 566, 'garbage truck': 569, 'dustcart': 569, 'gas pump': 571, 'gasoline pump': 571, 'petrol pump': 571, 'island dispenser': 571, 'golf ball': 574, 'parachute': 701, 'chute': 701}
                
                align_surrogate_testset size: 3925
                """
                
            elif args.surrogate_dataset == 'imagewoof':  
                transform_train = transforms.Compose([
                    transforms.RandomResizedCrop(size=224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

                transform_test = transforms.Compose([
                    transforms.Resize(size=224),
                    transforms.CenterCrop(size=224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                
                # test set
                surrogate_testset = datasets.ImageFolder(root=base_path +'/data/ImageWoof/imagewoof2/val', transform=transform_test)
                print(f'Test dataset size: {len(surrogate_testset)}')
                # print("Original surrogate_testset.classes:", surrogate_testset.classes)
                # print("Original surrogate_testset.class_to_idx:", surrogate_testset.class_to_idx)            
                with open(base_path +'/data/ImageNet2012/info/imagenet_class_index.json') as f:
                    class_idx = json.load(f)
                targetset_class_to_idx = {value[0]: int(key) for key, value in class_idx.items()}
                updated_classtoidx_surrogate_testset = AlignClassToIdx_ImageWoof(surrogate_testset, targetset_class_to_idx)   
                # print("updated_classtoidx_surrogate_testset.classes:", updated_classtoidx_surrogate_testset.classes)
                # print("updated_classtoidx_surrogate_testset.class_to_idx:", updated_classtoidx_surrogate_testset.class_to_idx)
                align_surrogate_testset = AlignedImageWoofDataset(root=base_path +'/data/ImageWoof/imagewoof2/val', transform=transform_test, class_to_idx=updated_classtoidx_surrogate_testset.class_to_idx)
                # print("align_surrogate_testset.classes:", align_surrogate_testset.classes)
                # print("align_surrogate_testset.class_to_idx:", align_surrogate_testset.class_to_idx)
                print(f'align_surrogate_testset size: {len(align_surrogate_testset.dataset)}')
                
                """ 
                args.surrogate_dataset is imagewoof
                Test dataset size: 3929
                
                Original surrogate_testset.classes: 
                ['n02086240', 'n02087394', 'n02088364', 'n02089973', 'n02093754', 'n02096294', 'n02099601', 'n02105641', 'n02111889', 'n02115641']
                
                Original surrogate_testset.class_to_idx: 
                {'n02086240': 0, 'n02087394': 1, 'n02088364': 2, 'n02089973': 3, 'n02093754': 4, 'n02096294': 5, 'n02099601': 6, 'n02105641': 7, 'n02111889': 8, 'n02115641': 9}
                
                updated_classtoidx_surrogate_testset.classes: 
                ['n02086240', 'n02087394', 'n02088364', 'n02089973', 'n02093754', 'n02096294', 'n02099601', 'n02105641', 'n02111889', 'n02115641']
                
                updated_classtoidx_surrogate_testset.class_to_idx: 
                {'n02086240': 155, 'n02087394': 159, 'n02088364': 162, 'n02089973': 167, 'n02093754': 182, 'n02096294': 193, 'n02099601': 207, 'n02105641': 229, 'n02111889': 258, 'n02115641': 273}
                
                align_surrogate_testset.classes: 
                ['n02086240', 'n02087394', 'n02088364', 'n02089973', 'n02093754', 'n02096294', 'n02099601', 'n02105641', 'n02111889', 'n02115641']
                
                align_surrogate_testset.class_to_idx: 
                {'n02086240': 155, 'n02087394': 159, 'n02088364': 162, 'n02089973': 167, 'n02093754': 182, 'n02096294': 193, 'n02099601': 207, 'n02105641': 229, 'n02111889': 258, 'n02115641': 273}
                
                align_surrogate_testset size: 3929
                """
                
            elif args.surrogate_dataset == 'tiny_imagenet':
                transform_train = transforms.Compose([
                    transforms.RandomResizedCrop(size=224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

                transform_test = transforms.Compose([
                    transforms.Resize(size=224),
                    transforms.CenterCrop(size=224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

                # test set
                # surrogate_testset = datasets.ImageFolder(root='/home/newdrive/huan1932/data/ImageWoof/imagewoof2/val', transform=transform_test)
                surrogate_trainset = datasets.ImageFolder(root=base_path +'/data/TinyImageNet/tiny-imagenet-200/train', transform=transform_train)
                surrogate_testset = TinyImageNetValDataset(root=base_path +'/data/TinyImageNet/tiny-imagenet-200/val', class_to_idx=surrogate_trainset.class_to_idx, transform=transform_test)
                print(f'Test dataset size: {len(surrogate_testset)}')
                # print("Original surrogate_testset.classes:", surrogate_testset.classes)
                # print("Original surrogate_testset.class_to_idx:", surrogate_testset.class_to_idx)            
                with open(base_path +'/data/ImageNet2012/info/imagenet_class_index.json') as f:
                    class_idx = json.load(f)
                targetset_class_to_idx = {value[0]: int(key) for key, value in class_idx.items()}
                updated_classtoidx_surrogate_testset = AlignClassToIdx_TinyImageNet(surrogate_testset, targetset_class_to_idx)   
                # print("updated_classtoidx_surrogate_testset.classes:", updated_classtoidx_surrogate_testset.classes)
                # print("updated_classtoidx_surrogate_testset.class_to_idx:", updated_classtoidx_surrogate_testset.class_to_idx)            
                align_surrogate_testset = AlignedTinyImageNetDataset(root=base_path +'/data/TinyImageNet/tiny-imagenet-200/val', transform=transform_test, class_to_idx=updated_classtoidx_surrogate_testset.class_to_idx)
                # print("align_surrogate_testset.classes:", align_surrogate_testset.classes)
                # print("align_surrogate_testset.class_to_idx:", align_surrogate_testset.class_to_idx)
                print(f'align_surrogate_testset size: {len(align_surrogate_testset.dataset)}')
                
                """ 
                args.surrogate_dataset is tiny_imagenet
                Test dataset size: 10000
                
                align_surrogate_testset.class_to_idx: {'n01443537': 1, 'n01629819': 25, 'n01641577': 30, 'n01644900': 32, 'n01698640': 50, 'n01742172': 61, 'n01768244': 69, 'n01770393': 71, 'n01774384': 75, 'n01774750': 76, 'n01784675': 79, 'n01855672': 99, 'n01882714': 105, 'n01910747': 107, 'n01917289': 109, 'n01944390': 113, 'n01945685': 114, 'n01950731': 115, 'n01983481': 122, 'n01984695': 123, 'n02002724': 128, 'n02056570': 145, 'n02058221': 146, 'n02074367': 149, 'n02085620': 151, 'n02094433': 187, 'n02099601': 207, 'n02099712': 208, 'n02106662': 235, 'n02113799': 267, 'n02123045': 281, 'n02123394': 283, 'n02124075': 285, 'n02125311': 286, 'n02129165': 291, 'n02132136': 294, 'n02165456': 301, 'n02190166': 308, 'n02206856': 309, 'n02226429': 311, 'n02231487': 313, 'n02233338': 314, 'n02236044': 315, 'n02268443': 319, 'n02279972': 323, 'n02281406': 325, 'n02321529': 329, 'n02364673': 338, 'n02395406': 341, 'n02403003': 345, 'n02410509': 347, 'n02415577': 349, 'n02423022': 353, 'n02437312': 354, 'n02480495': 365, 'n02481823': 367, 'n02486410': 372, 'n02504458': 386, 'n02509815': 387, 'n02666196': 398, 'n02669723': 400, 'n02699494': 406, 'n02730930': 411, 'n02769748': 414, 'n02788148': 421, 'n02791270': 424, 'n02793495': 425, 'n02795169': 427, 'n02802426': 430, 'n02808440': 435, 'n02814533': 436, 'n02814860': 437, 'n02815834': 438, 'n02823428': 440, 'n02837789': 445, 'n02841315': 447, 'n02843684': 448, 'n02883205': 457, 'n02892201': 458, 'n02906734': 462, 'n02909870': 463, 'n02917067': 466, 'n02927161': 467, 'n02948072': 470, 'n02950826': 471, 'n02963159': 474, 'n02977058': 480, 'n02988304': 485, 'n02999410': 488, 'n03014705': 492, 'n03026506': 496, 'n03042490': 500, 'n03085013': 508, 'n03089624': 509, 'n03100240': 511, 'n03126707': 517, 'n03160309': 525, 'n03179701': 526, 'n03201208': 532, 'n03250847': 542, 'n03255030': 543, 'n03355925': 557, 'n03388043': 562, 'n03393912': 565, 'n03400231': 567, 'n03404251': 568, 'n03424325': 570, 'n03444034': 573, 'n03447447': 576, 'n03544143': 604, 'n03584254': 605, 'n03599486': 612, 'n03617480': 614, 'n03637318': 619, 'n03649909': 621, 'n03662601': 625, 'n03670208': 627, 'n03706229': 635, 'n03733131': 645, 'n03763968': 652, 'n03770439': 655, 'n03796401': 675, 'n03804744': 677, 'n03814639': 678, 'n03837869': 682, 'n03838899': 683, 'n03854065': 687, 'n03891332': 704, 'n03902125': 707, 'n03930313': 716, 'n03937543': 720, 'n03970156': 731, 'n03976657': 733, 'n03977966': 734, 'n03980874': 735, 'n03983396': 737, 'n03992509': 739, 'n04008634': 744, 'n04023962': 747, 'n04067472': 758, 'n04070727': 760, 'n04074963': 761, 'n04099969': 765, 'n04118538': 768, 'n04133789': 774, 'n04146614': 779, 'n04149813': 781, 'n04179913': 786, 'n04251144': 801, 'n04254777': 806, 'n04259630': 808, 'n04265275': 811, 'n04275548': 815, 'n04285008': 817, 'n04311004': 821, 'n04328186': 826, 'n04356056': 837, 'n04366367': 839, 'n04371430': 842, 'n04376876': 845, 'n04398044': 849, 'n04399382': 850, 'n04417672': 853, 'n04456115': 862, 'n04465501': 866, 'n04486054': 873, 'n04487081': 874, 'n04501370': 877, 'n04507155': 879, 'n04532106': 887, 'n04532670': 888, 'n04540053': 890, 'n04560804': 899, 'n04562935': 900, 'n04596742': 909, 'n04597913': 910, 'n06596364': 917, 'n07579787': 923, 'n07583066': 924, 'n07614500': 928, 'n07615774': 929, 'n07695742': 932, 'n07711569': 935, 'n07715103': 938, 'n07720875': 945, 'n07734744': 947, 'n07747607': 950, 'n07749582': 951, 'n07753592': 954, 'n07768694': 957, 'n07871810': 962, 'n07873807': 963, 'n07875152': 964, 'n07920052': 967, 'n09193705': 970, 'n09246464': 972, 'n09256479': 973, 'n09332890': 975, 'n09428293': 978, 'n12267677': 988}
                
                align_surrogate_testset size: 10000
                """                        
            else:
                raise Exception('please input the valid clean surrogate dataset') 
        else:
            if args.target_dataset == 'cifar10':
                
                transform_test = transforms.Compose([
                    transforms.Resize(size=224),
                    transforms.CenterCrop(size=224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                target_testset = datasets.CIFAR10(root=base_path +'/data/CIFAR-10', train=False,transform=transform_test) 
            elif args.target_dataset == 'cifar100':
                transform_test = transforms.Compose([
                transforms.Resize(size=224),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                target_testset = datasets.CIFAR100(root=base_path +'/data/CIFAR-100', train=False,transform=transform_test) 
            if args.surrogate_dataset == 'imagenette':
                imagenette_testset = datasets.Imagenette(root= base_path +'/data/ImageNette', split='val')      
                updated_classtoidx_imagenette_testset = AlignCifarto_Imagenette(imagenette_testset,target_testset.class_to_idx,args.target_dataset)            
                filtered_testset_label_idx = set(updated_classtoidx_imagenette_testset.class_to_idx.values())
                
                print("filtered_testset_label_idx:", filtered_testset_label_idx)
                """ 
                filtered_testset_label_idx: 
                {0, 482, 491, 569, 497, 566, 217, 571, 701, 574}
                """
            elif args.surrogate_dataset == 'imagewoof':
                imagewoof_testset = datasets.ImageFolder(root=base_path +'/data/ImageWoof/imagewoof2/val')         
                updated_classtoidx_imagewoof_testset = AlignCifarto_ImageWoof(imagewoof_testset, args.target_dataset)   
                filtered_testset_label_idx = set(updated_classtoidx_imagewoof_testset.class_to_idx.values())
                print("filtered_testset_label_idx:", filtered_testset_label_idx)   
                """ 
                filtered_dataset_label_idx: {193, 162, 258, 229, 167, 207, 273, 182, 155, 159}
              
                """  
            
            align_surrogate_testset = FilteredDataset(target_testset, filtered_testset_label_idx,args.target_dataset)
             
    if adv == True:
        if args.taskmode == 'transfer_attack':
            if args.surrogate_dataset == 'imagenette':
                
                surrogate_testset = datasets.Imagenette(root=base_path +'/data/ImageNette', split='val')
                print(f'Test dataset size: {len(surrogate_testset)}')
                original_class_to_idx = copy.deepcopy(surrogate_testset.class_to_idx)
                print("original_class_to_idx:", original_class_to_idx)  
                imagenet_testset = datasets.ImageNet(root=base_path +'/data/ImageNet2012', split='val')   
                    
                if args.all:
                    updated_classtoidx_surrogate_testset = AlignClassToIdx_ImageNette(surrogate_testset, imagenet_testset.class_to_idx) 
                else:    
                    updated_classtoidx_surrogate_testset = AlignClassToIdx_ImageNette2(surrogate_testset, imagenet_testset.class_to_idx,args.target_dataset)   
                updated_classtoidx = updated_classtoidx_surrogate_testset.class_to_idx
                # print("updated_classtoidx:", updated_classtoidx)                

                """ 
                original_class_to_idx: 
                {'tench': 0, 'Tinca tinca': 0, 'English springer': 1, 'English springer spaniel': 1, 'cassette player': 2, 'chain saw': 3, 'chainsaw': 3, 'church': 4, 'church building': 4, 'French horn': 5, 'horn': 5, 'garbage truck': 6, 'dustcart': 6, 'gas pump': 7, 'gasoline pump': 7, 'petrol pump': 7, 'island dispenser': 7, 'golf ball': 8, 'parachute': 9, 'chute': 9}
                
                updated_classtoidx: {'tench': 0, 'Tinca tinca': 0, 'English springer': 217, 'English springer spaniel': 217, 'cassette player': 482, 'chain saw': 491, 'chainsaw': 491, 'church': 497, 'church building': 497, 'French horn': 566, 'horn': 566, 'garbage truck': 569, 'dustcart': 569, 'gas pump': 571, 'gasoline pump': 571, 'petrol pump': 571, 'island dispenser': 571, 'golf ball': 574, 'parachute': 701, 'chute': 701}
                
                
                original adv_labels: tensor([0, 0, 0,  ..., 9, 9, 9], device='cuda:0')
                class_map: {0: 0, 1: 217, 2: 482, 3: 491, 4: 497, 5: 566, 6: 569, 7: 571, 8: 574, 9: 701}
                mapped adv_labels: tensor([  0,   0,   0,  ..., 701, 701, 701])
                """
                
            elif args.surrogate_dataset == 'imagewoof':  
                # test set
                surrogate_testset = datasets.ImageFolder(root=base_path +'/data/ImageWoof/imagewoof2/val')
                print(f'Test dataset size: {len(surrogate_testset)}')
                original_class_to_idx = copy.deepcopy(surrogate_testset.class_to_idx)
                print("original_class_to_idx:", original_class_to_idx)           
                with open(base_path +'/data/ImageNet2012/info/imagenet_class_index.json') as f:
                    class_idx = json.load(f)
                targetset_class_to_idx = {value[0]: int(key) for key, value in class_idx.items()}
                updated_classtoidx_surrogate_testset = AlignClassToIdx_ImageWoof(surrogate_testset, targetset_class_to_idx)   
                updated_classtoidx = updated_classtoidx_surrogate_testset.class_to_idx
                # print("updated_classtoidx:", updated_classtoidx)        
                """ 
                class_map: {0: 155, 1: 159, 2: 162, 3: 167, 4: 182, 5: 193, 6: 207, 7: 229, 8: 258, 9: 273}
                """        
                
            elif args.surrogate_dataset == 'tiny_imagenet':
                surrogate_trainset = datasets.ImageFolder(root=base_path +'/data/TinyImageNet/tiny-imagenet-200/train')
                surrogate_testset = TinyImageNetValDataset(root=base_path +'/data/TinyImageNet/tiny-imagenet-200/val', class_to_idx=surrogate_trainset.class_to_idx)
                original_class_to_idx = copy.deepcopy(surrogate_testset.class_to_idx)
                print("original_class_to_idx:", original_class_to_idx)   
                
                print(f'Test dataset size: {len(surrogate_testset)}')
                # print("Original surrogate_testset.classes:", surrogate_testset.classes)
                # print("Original surrogate_testset.class_to_idx:", surrogate_testset.class_to_idx)            
                with open(base_path +'/data/ImageNet2012/info/imagenet_class_index.json') as f:
                    class_idx = json.load(f)
                targetset_class_to_idx = {value[0]: int(key) for key, value in class_idx.items()}
                updated_classtoidx_surrogate_testset = AlignClassToIdx_TinyImageNet(surrogate_testset, targetset_class_to_idx)   
                updated_classtoidx = updated_classtoidx_surrogate_testset.class_to_idx
                # print("updated_classtoidx:", updated_classtoidx)      
                
                """ 
                class_map: 
                {0: 1, 1: 25, 2: 30, 3: 32, 4: 50, 5: 61, 6: 69, 7: 71, 8: 75, 9: 76, 10: 79, 11: 99, 12: 105, 13: 107, 14: 109, 15: 113, 16: 114, 17: 115, 18: 122, 19: 123, 20: 128, 21: 145, 22: 146, 23: 149, 24: 151, 25: 187, 26: 207, 27: 208, 28: 235, 29: 267, 30: 281, 31: 283, 32: 285, 33: 286, 34: 291, 35: 294, 36: 301, 37: 308, 38: 309, 39: 311, 40: 313, 41: 314, 42: 315, 43: 319, 44: 323, 45: 325, 46: 329, 47: 338, 48: 341, 49: 345, 50: 347, 51: 349, 52: 353, 53: 354, 54: 365, 55: 367, 56: 372, 57: 386, 58: 387, 59: 398, 60: 400, 61: 406, 62: 411, 63: 414, 64: 421, 65: 424, 66: 425, 67: 427, 68: 430, 69: 435, 70: 436, 71: 437, 72: 438, 73: 440, 74: 445, 75: 447, 76: 448, 77: 457, 78: 458, 79: 462, 80: 463, 81: 466, 82: 467, 83: 470, 84: 471, 85: 474, 86: 480, 87: 485, 88: 488, 89: 492, 90: 496, 91: 500, 92: 508, 93: 509, 94: 511, 95: 517, 96: 525, 97: 526, 98: 532, 99: 542, 100: 543, 101: 557, 102: 562, 103: 565, 104: 567, 105: 568, 106: 570, 107: 573, 108: 576, 109: 604, 110: 605, 111: 612, 112: 614, 113: 619, 114: 621, 115: 625, 116: 627, 117: 635, 118: 645, 119: 652, 120: 655, 121: 675, 122: 677, 123: 678, 124: 682, 125: 683, 126: 687, 127: 704, 128: 707, 129: 716, 130: 720, 131: 731, 132: 733, 133: 734, 134: 735, 135: 737, 136: 739, 137: 744, 138: 747, 139: 758, 140: 760, 141: 761, 142: 765, 143: 768, 144: 774, 145: 779, 146: 781, 147: 786, 148: 801, 149: 806, 150: 808, 151: 811, 152: 815, 153: 817, 154: 821, 155: 826, 156: 837, 157: 839, 158: 842, 159: 845, 160: 849, 161: 850, 162: 853, 163: 862, 164: 866, 165: 873, 166: 874, 167: 877, 168: 879, 169: 887, 170: 888, 171: 890, 172: 899, 173: 900, 174: 909, 175: 910, 176: 917, 177: 923, 178: 924, 179: 928, 180: 929, 181: 932, 182: 935, 183: 938, 184: 945, 185: 947, 186: 950, 187: 951, 188: 954, 189: 957, 190: 962, 191: 963, 192: 964, 193: 967, 194: 970, 195: 972, 196: 973, 197: 975, 198: 978, 199: 988}
                """
                

            else:
                raise Exception('please input the valid adv surrogate dataset') 
            
        elif args.taskmode == 'transfer_attack_target':
            if args.surrogate_dataset == 'imagenette':

                surrogate_testset = datasets.Imagenette(root=base_path +'/data/ImageNette', split='val')
                print(f'Test dataset size: {len(surrogate_testset)}')
                original_class_to_idx = copy.deepcopy(surrogate_testset.class_to_idx)
                # print("original_class_to_idx:", original_class_to_idx)   
                if args.target_dataset == 'cifar10':
                    target_testset = datasets.CIFAR10(root=base_path +'/data/CIFAR-10', train=False) 
                elif args.target_dataset == 'cifar100':
                    target_testset = datasets.CIFAR100(root=base_path +'/data/CIFAR-100', train=False) 

                updated_classtoidx_surrogate_testset = AlignCifarto_Imagenette(surrogate_testset,target_testset.class_to_idx, args.target_dataset)      
                updated_classtoidx = updated_classtoidx_surrogate_testset.class_to_idx
                # print("updated_classtoidx:", updated_classtoidx)                

                """ 
                original_class_to_idx: 
                {'tench': 0, 'Tinca tinca': 0, 'English springer': 1, 'English springer spaniel': 1, 'cassette player': 2, 'chain saw': 3, 'chainsaw': 3, 'church': 4, 'church building': 4, 'French horn': 5, 'horn': 5, 'garbage truck': 6, 'dustcart': 6, 'gas pump': 7, 'gasoline pump': 7, 'petrol pump': 7, 'island dispenser': 7, 'golf ball': 8, 'parachute': 9, 'chute': 9}
                
                updated_classtoidx: {'tench': 0, 'Tinca tinca': 0, 'English springer': 217, 'English springer spaniel': 217, 'cassette player': 482, 'chain saw': 491, 'chainsaw': 491, 'church': 497, 'church building': 497, 'French horn': 566, 'horn': 566, 'garbage truck': 569, 'dustcart': 569, 'gas pump': 571, 'gasoline pump': 571, 'petrol pump': 571, 'island dispenser': 571, 'golf ball': 574, 'parachute': 701, 'chute': 701}
                
                
                original adv_labels: tensor([0, 0, 0,  ..., 9, 9, 9], device='cuda:0')
                class_map: {0: 0, 1: 217, 2: 482, 3: 491, 4: 497, 5: 566, 6: 569, 7: 571, 8: 574, 9: 701}
                mapped adv_labels: tensor([  0,   0,   0,  ..., 701, 701, 701])
                """
                
            elif args.surrogate_dataset == 'imagewoof':  
                # test set
                surrogate_testset = datasets.ImageFolder(root=base_path +'/data/ImageWoof/imagewoof2/val')
                print(f'Test dataset size: {len(surrogate_testset)}')
                original_class_to_idx = copy.deepcopy(surrogate_testset.class_to_idx)
                print("original_class_to_idx:", original_class_to_idx)           
                updated_classtoidx_surrogate_testset =  AlignCifarto_ImageWoof(surrogate_testset,args.target_dataset) 
                updated_classtoidx = updated_classtoidx_surrogate_testset.class_to_idx
                # print("updated_classtoidx:", updated_classtoidx)        
                """ 
                class_map: {0: 155, 1: 159, 2: 162, 3: 167, 4: 182, 5: 193, 6: 207, 7: 229, 8: 258, 9: 273}
                """        
        print("args.surrogate_advdataset_x_path:",args.surrogate_advdataset_x_path)
        print("args.surrogate_advdataset_y_path:",args.surrogate_advdataset_y_path)            
        adv_images = torch.load(args.surrogate_advdataset_x_path)  # 替换为你的对抗性图像数据的路径
        adv_labels = torch.load(args.surrogate_advdataset_y_path)  # 替换为你的对抗性标签数据的路径
        print("original adv_labels:",adv_labels)
        class_map = {}
        if args.taskmode =='transfer_attack_target':
            classes = []
            for k,v in updated_classtoidx.items():
                classes.append(k)
            for key, value in original_class_to_idx.items():
                if key in classes:
                    class_map[value] = updated_classtoidx[key]
        elif args.taskmode == 'transfer_attack':
            if args.all:
                for key, value in original_class_to_idx.items():
                    # print(f"key-{key}: value-{value}")
                    updated_value = updated_classtoidx[key]
                    # print("updated_value:",updated_value)
                    class_map[value] = updated_value
                    # print(f"class_map[{value}] = {updated_value}")
            else: 
                classes = []
                for k,v in updated_classtoidx.items():
                    classes.append(k)
                for key, value in original_class_to_idx.items():
                    if key in classes:
                        class_map[value] = updated_classtoidx[key]
        print("class_map:", class_map)
        mask = torch.tensor([label.item() in class_map for label in adv_labels])
        adv_labels = torch.tensor([class_map[label.item()] for label in adv_labels if label.item() in class_map])
        print("mapped adv_labels:",adv_labels)
        align_surrogate_testset = TensorDataset(adv_images[mask], adv_labels)
        
                    
    # trainset and val set
    align_surrogate_trainset = None
    align_surrogate_valset = None      
        
    return align_surrogate_trainset, align_surrogate_valset, align_surrogate_testset

def get_target_dataset(args, clean=True, adv=False):
    
    print(f'args.target_dataset is {args.target_dataset}')
    
    if clean == True:    
        if args.target_dataset == 'cifar10':
            
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(size=224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            transform_test = transforms.Compose([
                transforms.Resize(size=224),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            target_testset = datasets.CIFAR10(root=base_path +'/data/CIFAR-10', train=False, transform=transform_test)    
            target_trainset = datasets.CIFAR10(root=base_path +'/data/CIFAR-10', train=True, transform=transform_train)    

            train_size = int(0.8 * len(target_trainset))
            val_size = len(target_trainset) - train_size

            target_trainset, target_valset = torch.utils.data.random_split(target_trainset, [train_size, val_size])
            print(f'Splitted test dataset size: {len(target_testset)}')            
            print(f'Splitted training dataset size: {len(target_trainset)}')
            print(f'Splitted validation dataset size: {len(target_valset)}')

            original_class_to_idx = target_testset.class_to_idx
            print("CIFAR-10 original_class_to_idx:", original_class_to_idx)       
            """ 
            CIFAR-10 original_class_to_idx: 
            {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
            """       

        elif args.target_dataset == 'cifar100':
            
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(size=224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            transform_test = transforms.Compose([
                transforms.Resize(size=224),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            target_testset = datasets.CIFAR100(root=base_path +'/data/CIFAR-100', train=False, transform=transform_test)    
            target_trainset = datasets.CIFAR100(root=base_path +'/data/CIFAR-100', train=True, transform=transform_train)    

            train_size = int(0.8 * len(target_trainset))
            val_size = len(target_trainset) - train_size

            target_trainset, target_valset = torch.utils.data.random_split(target_trainset, [train_size, val_size])
            print(f'Splitted test dataset size: {len(target_testset)}')            
            print(f'Splitted training dataset size: {len(target_trainset)}')
            print(f'Splitted validation dataset size: {len(target_valset)}')
            
            original_class_to_idx = target_testset.class_to_idx
            print("CIFAR-100 original_class_to_idx:", original_class_to_idx)      
            
            """ 
            CIFAR-100 original_class_to_idx: 
            {'apple': 0, 'aquarium_fish': 1, 'baby': 2, 'bear': 3, 'beaver': 4, 'bed': 5, 'bee': 6, 'beetle': 7, 'bicycle': 8, 'bottle': 9, 'bowl': 10, 'boy': 11, 'bridge': 12, 'bus': 13, 'butterfly': 14, 'camel': 15, 'can': 16, 'castle': 17, 'caterpillar': 18, 'cattle': 19, 'chair': 20, 'chimpanzee': 21, 'clock': 22, 'cloud': 23, 'cockroach': 24, 'couch': 25, 'crab': 26, 'crocodile': 27, 'cup': 28, 'dinosaur': 29, 'dolphin': 30, 'elephant': 31, 'flatfish': 32, 'forest': 33, 'fox': 34, 'girl': 35, 'hamster': 36, 'house': 37, 'kangaroo': 38, 'keyboard': 39, 'lamp': 40, 'lawn_mower': 41, 'leopard': 42, 'lion': 43, 'lizard': 44, 'lobster': 45, 'man': 46, 'maple_tree': 47, 'motorcycle': 48, 'mountain': 49, 'mouse': 50, 'mushroom': 51, 'oak_tree': 52, 'orange': 53, 'orchid': 54, 'otter': 55, 'palm_tree': 56, 'pear': 57, 'pickup_truck': 58, 'pine_tree': 59, 'plain': 60, 'plate': 61, 'poppy': 62, 'porcupine': 63, 'possum': 64, 'rabbit': 65, 'raccoon': 66, 'ray': 67, 'road': 68, 'rocket': 69, 'rose': 70, 'sea': 71, 'seal': 72, 'shark': 73, 'shrew': 74, 'skunk': 75, 'skyscraper': 76, 'snail': 77, 'snake': 78, 'spider': 79, 'squirrel': 80, 'streetcar': 81, 'sunflower': 82, 'sweet_pepper': 83, 'table': 84, 'tank': 85, 'telephone': 86, 'television': 87, 'tiger': 88, 'tractor': 89, 'train': 90, 'trout': 91, 'tulip': 92, 'turtle': 93, 'wardrobe': 94, 'whale': 95, 'willow_tree': 96, 'wolf': 97, 'woman': 98, 'worm': 99}
            """                               
        else:
            raise Exception('please input the valid clean target dataset') 
            
    return target_trainset, target_valset, target_testset
    
def get_source_dataloader(args):
    
    source_trainset, source_valset, source_test_dataset = get_source_dataset(args)
    source_train_loader = DataLoader(source_trainset, batch_size=args.batch_size, shuffle=True)
    source_val_loader = DataLoader(source_valset, batch_size=args.batch_size, shuffle=False)    
    source_test_loader = DataLoader(source_test_dataset, batch_size=args.batch_size, shuffle=False)
    
    return source_train_loader, source_val_loader, source_test_loader

def get_surrogate_dataloader(args, clean=False, adv=False):
    surrogate_trainset, surrogate_valset, surrogate_testset = get_surrogate_dataset(args=args, clean=clean, adv=adv)
    
    if surrogate_trainset == None:
        surrogate_train_loader = None
    else:
        surrogate_train_loader = DataLoader(surrogate_trainset, batch_size=args.batch_size, shuffle=True)
        
    if surrogate_valset == None:
        surrogate_val_loader = None
    else:
        surrogate_val_loader = DataLoader(surrogate_valset, batch_size=args.batch_size, shuffle=False)    

    if surrogate_testset == None:
        surrogate_test_loader = None
    else:
        surrogate_test_loader = DataLoader(surrogate_testset, batch_size=args.batch_size, shuffle=False)
    
    return surrogate_train_loader, surrogate_val_loader, surrogate_test_loader    

def get_align_surrogate_dataloader(args, clean=False, adv=False):
    
    align_surrogate_trainset, align_surrogate_valset, align_surrogate_testset = get_align_surrogate_dataset(args=args, clean=clean, adv=adv)
    
    if align_surrogate_trainset == None:
        align_surrogate_train_loader = None
    else:
        align_surrogate_train_loader = DataLoader(align_surrogate_trainset, batch_size=args.batch_size, shuffle=True)
        
    if align_surrogate_valset == None:
        align_surrogate_val_loader = None    
    else:
        align_surrogate_val_loader = DataLoader(align_surrogate_valset, batch_size=args.batch_size, shuffle=False) 
        
    if align_surrogate_testset == None:
        align_surrogate_test_loader = None
    else:  
        align_surrogate_test_loader = DataLoader(align_surrogate_testset, batch_size=args.batch_size, shuffle=False)
    
    return align_surrogate_train_loader, align_surrogate_val_loader, align_surrogate_test_loader    

 
def get_filter_source_dataloader(source_test_loader, args, filter_target):
    
    if args.taskmode == 'transfer_attack':
        if args.all:
            if filter_target == 'tiny_imagenet':
                tinyimagenet_trainset = datasets.ImageFolder(root=base_path +'/data/TinyImageNet/tiny-imagenet-200/train')
                tinyimagenet_testset = TinyImageNetValDataset(root=base_path +'/data/TinyImageNet/tiny-imagenet-200/val', class_to_idx=tinyimagenet_trainset.class_to_idx)       
                with open(base_path +'/data/ImageNet2012/info/imagenet_class_index.json') as f:
                    class_idx = json.load(f)
                targetset_class_to_idx = {value[0]: int(key) for key, value in class_idx.items()}
                updated_classtoidx_tinyimagenet_testset = AlignClassToIdx_TinyImageNet(tinyimagenet_testset, targetset_class_to_idx)   
                filtered_testset_label_idx = set(updated_classtoidx_tinyimagenet_testset.class_to_idx.values())
                # print("filtered_testset_label_idx:", filtered_testset_label_idx)
                """ 
                filtered_testset_label_idx: 
                {1, 517, 525, 526, 532, 25, 30, 542, 32, 543, 557, 50, 562, 565, 567, 568, 570, 61, 573, 576, 69, 71, 75, 76, 79, 604, 605, 99, 612, 614, 105, 107, 619, 109, 621, 113, 114, 115, 625, 627, 122, 123, 635, 128, 645, 652, 655, 145, 146, 149, 151, 675, 677, 678, 682, 683, 687, 187, 704, 707, 716, 207, 208, 720, 731, 733, 734, 735, 737, 739, 744, 235, 747, 758, 760, 761, 765, 768, 774, 267, 779, 781, 786, 281, 283, 285, 286, 801, 291, 294, 806, 808, 811, 301, 815, 817, 308, 309, 821, 311, 313, 314, 315, 826, 319, 323, 325, 837, 839, 329, 842, 845, 849, 338, 850, 341, 853, 345, 347, 349, 862, 353, 354, 866, 873, 874, 365, 877, 367, 879, 372, 887, 888, 890, 386, 387, 899, 900, 909, 398, 910, 400, 917, 406, 411, 923, 924, 414, 928, 929, 932, 421, 935, 424, 425, 938, 427, 430, 945, 435, 436, 437, 438, 947, 440, 950, 951, 954, 445, 957, 447, 448, 962, 963, 964, 967, 457, 458, 970, 972, 973, 462, 463, 975, 466, 467, 978, 470, 471, 474, 988, 480, 485, 488, 492, 496, 500, 508, 509, 511}
                """
            elif filter_target == 'imagenette':
                imagenette_testset = datasets.Imagenette(root= base_path +'/data/ImageNette', split='val')
                imagenet_testset = datasets.ImageNet(root=base_path +'/data/ImageNet2012', split='val')       
                updated_classtoidx_imagenette_testset = AlignClassToIdx_ImageNette(imagenette_testset, imagenet_testset.class_to_idx)            
                filtered_testset_label_idx = set(updated_classtoidx_imagenette_testset.class_to_idx.values())
                # print("filtered_testset_label_idx:", filtered_testset_label_idx)
                """ 
                filtered_testset_label_idx: 
                {0, 482, 491, 569, 497, 566, 217, 571, 701, 574}
                """
            elif filter_target == 'imagewoof':
                imagewoof_testset = datasets.ImageFolder(root=base_path +'/data/ImageWoof/imagewoof2/val')         
                with open(base_path +'/data/ImageNet2012/info/imagenet_class_index.json') as f:
                    class_idx = json.load(f)
                targetset_class_to_idx = {value[0]: int(key) for key, value in class_idx.items()}
                updated_classtoidx_imagewoof_testset = AlignClassToIdx_ImageWoof(imagewoof_testset, targetset_class_to_idx)   
                filtered_testset_label_idx = set(updated_classtoidx_imagewoof_testset.class_to_idx.values())
                # print("filtered_testset_label_idx:", filtered_testset_label_idx)   
                """ 
                filtered_dataset_label_idx: {193, 162, 258, 229, 167, 207, 273, 182, 155, 159}
                """  
        else:
            if filter_target == 'tiny_imagenet':
                tinyimagenet_trainset = datasets.ImageFolder(root=base_path +'/data/TinyImageNet/tiny-imagenet-200/train')
                tinyimagenet_testset = TinyImageNetValDataset(root=base_path +'/data/TinyImageNet/tiny-imagenet-200/val', class_to_idx=tinyimagenet_trainset.class_to_idx)       
                with open(base_path +'/data/ImageNet2012/info/imagenet_class_index.json') as f:
                    class_idx = json.load(f)
                targetset_class_to_idx = {value[0]: int(key) for key, value in class_idx.items()}
                updated_classtoidx_tinyimagenet_testset = AlignClassToIdx_TinyImageNet(tinyimagenet_testset, targetset_class_to_idx)   
                filtered_testset_label_idx = set(updated_classtoidx_tinyimagenet_testset.class_to_idx.values())
                # print("filtered_testset_label_idx:", filtered_testset_label_idx)
                """ 
                filtered_testset_label_idx: 
                {1, 517, 525, 526, 532, 25, 30, 542, 32, 543, 557, 50, 562, 565, 567, 568, 570, 61, 573, 576, 69, 71, 75, 76, 79, 604, 605, 99, 612, 614, 105, 107, 619, 109, 621, 113, 114, 115, 625, 627, 122, 123, 635, 128, 645, 652, 655, 145, 146, 149, 151, 675, 677, 678, 682, 683, 687, 187, 704, 707, 716, 207, 208, 720, 731, 733, 734, 735, 737, 739, 744, 235, 747, 758, 760, 761, 765, 768, 774, 267, 779, 781, 786, 281, 283, 285, 286, 801, 291, 294, 806, 808, 811, 301, 815, 817, 308, 309, 821, 311, 313, 314, 315, 826, 319, 323, 325, 837, 839, 329, 842, 845, 849, 338, 850, 341, 853, 345, 347, 349, 862, 353, 354, 866, 873, 874, 365, 877, 367, 879, 372, 887, 888, 890, 386, 387, 899, 900, 909, 398, 910, 400, 917, 406, 411, 923, 924, 414, 928, 929, 932, 421, 935, 424, 425, 938, 427, 430, 945, 435, 436, 437, 438, 947, 440, 950, 951, 954, 445, 957, 447, 448, 962, 963, 964, 967, 457, 458, 970, 972, 973, 462, 463, 975, 466, 467, 978, 470, 471, 474, 988, 480, 485, 488, 492, 496, 500, 508, 509, 511}
                """
            elif filter_target == 'imagenette':
                imagenette_testset = datasets.Imagenette(root= base_path +'/data/ImageNette', split='val')
                imagenet_testset = datasets.ImageNet(root=base_path +'/data/ImageNet2012', split='val')       
                updated_classtoidx_imagenette_testset = AlignClassToIdx_ImageNette2(imagenette_testset, imagenet_testset.class_to_idx,args.target_dataset)            
                filtered_testset_label_idx = set(updated_classtoidx_imagenette_testset.class_to_idx.values())
                # print("filtered_testset_label_idx:", filtered_testset_label_idx)
                """ 
                filtered_testset_label_idx: 
                {0, 482, 491, 569, 497, 566, 217, 571, 701, 574}
                """
            elif filter_target == 'imagewoof':
                imagewoof_testset = datasets.ImageFolder(root=base_path +'/data/ImageWoof/imagewoof2/val')         
                with open(base_path +'/data/ImageNet2012/info/imagenet_class_index.json') as f:
                    class_idx = json.load(f)
                targetset_class_to_idx = {value[0]: int(key) for key, value in class_idx.items()}
                updated_classtoidx_imagewoof_testset = AlignClassToIdx_ImageWoof(imagewoof_testset, targetset_class_to_idx)   
                filtered_testset_label_idx = set(updated_classtoidx_imagewoof_testset.class_to_idx.values())
                # print("filtered_testset_label_idx:", filtered_testset_label_idx)   
                """ 
                filtered_dataset_label_idx: {193, 162, 258, 229, 167, 207, 273, 182, 155, 159}
                """  

    elif args.taskmode == 'transfer_attack_target':
        if filter_target == 'imagenette':
            imagenette_testset = datasets.Imagenette(root= base_path +'/data/ImageNette', split='val')      
            updated_classtoidx_imagenette_testset = AlignCifarto_Imagenette(imagenette_testset,args.target_dataset)            
            filtered_testset_label_idx = set(updated_classtoidx_imagenette_testset.class_to_idx.values())
            
            print("filtered_testset_label_idx:", filtered_testset_label_idx)
            """ 
            filtered_testset_label_idx: 
            {0, 482, 491, 569, 497, 566, 217, 571, 701, 574}
            """
        elif filter_target == 'imagewoof':
            imagewoof_testset = datasets.ImageFolder(root=base_path +'/data/ImageWoof/imagewoof2/val')         
            updated_classtoidx_imagewoof_testset = AlignCifarto_ImageWoof(imagewoof_testset, args.target_dataset)   
            filtered_testset_label_idx = set(updated_classtoidx_imagewoof_testset.class_to_idx.values())
            print("filtered_testset_label_idx:", filtered_testset_label_idx)   
            """ 
            filtered_dataset_label_idx: {193, 162, 258, 229, 167, 207, 273, 182, 155, 159}
            """  
    print('here1') 
    filtered_testset = FilteredDataset(source_test_loader.dataset, filtered_testset_label_idx,args.target_dataset)
    filtered_testloader = DataLoader(filtered_testset, batch_size=args.batch_size, shuffle=False)  
    return filtered_testloader
 
 
def get_target_dataloader(args, clean=True, adv=False):
    if args.adapt_target_method =='card':
        args.is_sample = True
        if args.target_dataset == 'cifar10':
            target_trainset, target_valset, target_testset = get_cifar10_dataloaders_sample(args=args,k=args.nce_k)
            return target_trainset, target_valset, target_testset
        elif args.target_dataset == 'cifar100':
            target_trainset, target_valset, target_testset = get_cifar100_dataloaders_sample(args=args,k=args.nce_k)
            return target_trainset, target_valset, target_testset
    else:
        target_trainset, target_valset, target_testset = get_target_dataset(args=args, clean=clean, adv=adv)
    
    if target_trainset == None:
        target_train_loader = None
    else:
        target_train_loader = DataLoader(target_trainset, batch_size=args.batch_size, shuffle=True)
        
    if target_valset == None:
        target_val_loader = None
    else:
        target_val_loader = DataLoader(target_valset, batch_size=args.batch_size, shuffle=False)    

    if target_testset == None:
        target_test_loader = None
    else:
        target_test_loader = DataLoader(target_testset, batch_size=args.batch_size, shuffle=False)
    
    return target_train_loader, target_val_loader, target_test_loader    