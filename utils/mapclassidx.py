
def MapClassIdx(dataset, imagenet_class_to_idx):
    for class_tuple in dataset.classes:
        for class_name in class_tuple:
            if class_name in imagenet_class_to_idx:
                dataset.class_to_idx[class_name] = imagenet_class_to_idx[class_name]
    return dataset

def AlignClassToIdx_ImageNette(sourceset, targetset_class_to_idx):
    for class_tuple in sourceset.classes:
        for class_name in class_tuple:
            if class_name in targetset_class_to_idx:
                sourceset.class_to_idx[class_name] = targetset_class_to_idx[class_name]
    return sourceset

def AlignClassToIdx_ImageWoof(sourceset, targetset_class_to_idx):
    for class_identifier in sourceset.classes:
        if class_identifier in targetset_class_to_idx:
            sourceset.class_to_idx[class_identifier] = targetset_class_to_idx[class_identifier]
    return sourceset

def AlignClassToIdx_ImageNette2(sourceset, targetset_class_to_idx,targetset):
    if targetset == 'cifar10':
        allowedclass = ["English springer",
                "English springer spaniel",
                'garbage truck',
                'dustcart']
    else:
        allowedclass = [
            "English springer",
            "English springer spaniel",
            'garbage truck',
            'dustcart',
            'church',
            'church building',
            'tench',
            'Tinca tinca'
        ]
    filteridx = {}
    for class_tuple in sourceset.classes:
        for class_name in class_tuple:
            if class_name in targetset_class_to_idx and class_name in allowedclass:
                filteridx[class_name] = targetset_class_to_idx[class_name]
    sourceset.class_to_idx = filteridx
    return sourceset





def AlignCifarto_ImageWoof(sourceset, targetset):
    if targetset =='cifar10':
        for i in sourceset.classes:
            sourceset.class_to_idx[i] = 5
    elif targetset == 'cifar100':
        for i in sourceset.classes:
            sourceset.class_to_idx[i] = 97
    return sourceset
#{'tench': 0, 'Tinca tinca': 0, 'English springer': 1, 'English springer spaniel': 1, 'cassette player': 2, 'chain saw': 3, 'chainsaw': 3, 'church': 4, 'church building': 4, 'French horn': 5, 'horn': 5, 'garbage truck': 6, 'dustcart': 6, 'gas pump': 7, 'gasoline pump': 7, 'petrol pump': 7, 'island dispenser': 7, 'golf ball': 8, 'parachute': 9, 'chute': 9}


#{'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}



#{'apple': 0, 'aquarium_fish': 1, 'baby': 2, 'bear': 3, 'beaver': 4, 'bed': 5, 'bee': 6, 'beetle': 7, 'bicycle': 8, 'bottle': 9, 'bowl': 10, 'boy': 11, 'bridge': 12, 'bus': 13, 'butterfly': 14, 'camel': 15, 'can': 16, 'castle': 17, 'caterpillar': 18, 'cattle': 19, 'chair': 20, 'chimpanzee': 21, 'clock': 22, 'cloud': 23, 'cockroach': 24, 'couch': 25, 'crab': 26, 'crocodile': 27, 'cup': 28, 'dinosaur': 29, 'dolphin': 30, 'elephant': 31, 'flatfish': 32, 'forest': 33, 'fox': 34, 'girl': 35, 'hamster': 36, 'house': 37, 'kangaroo': 38, 'keyboard': 39, 'lamp': 40, 'lawn_mower': 41, 'leopard': 42, 'lion': 43, 'lizard': 44, 'lobster': 45, 'man': 46, 'maple_tree': 47, 'motorcycle': 48, 'mountain': 49, 'mouse': 50, 'mushroom': 51, 'oak_tree': 52, 'orange': 53, 'orchid': 54, 'otter': 55, 'palm_tree': 56, 'pear': 57, 'pickup_truck': 58, 'pine_tree': 59, 'plain': 60, 'plate': 61, 'poppy': 62, 'porcupine': 63, 'possum': 64, 'rabbit': 65, 'raccoon': 66, 'ray': 67, 'road': 68, 'rocket': 69, 'rose': 70, 'sea': 71, 'seal': 72, 'shark': 73, 'shrew': 74, 'skunk': 75, 'skyscraper': 76, 'snail': 77, 'snake': 78, 'spider': 79, 'squirrel': 80, 'streetcar': 81, 'sunflower': 82, 'sweet_pepper': 83, 'table': 84, 'tank': 85, 'telephone': 86, 'television': 87, 'tiger': 88, 'tractor': 89, 'train': 90, 'trout': 91, 'tulip': 92, 'turtle': 93, 'wardrobe': 94, 'whale': 95, 'willow_tree': 96, 'wolf': 97, 'woman': 98, 'worm': 99}


def AlignCifarto_Imagenette(sourceset,targetset_class_to_idx, targetset):
    print("targetset_class_to_idx:", targetset_class_to_idx)  
    if targetset == 'cifar10':
        # Define the mapping for CIFAR-10 target set
        allowed_classes = {
            "English springer": 5,
            "English springer spaniel": 5,
            'garbage truck': 9,
            'dustcart': 9
        }
    elif targetset == 'cifar100':
        # Define the mapping for CIFAR-100 target set
        allowed_classes = {
            "English springer": 97,
            "English springer spaniel": 97,
            'garbage truck': 58,
            'dustcart': 58,
            'church': 17,
            'church building': 17,
            'tench': 91,
            'Tinca tinca': 91
        }
    else:
        raise ValueError(f"Unsupported targetset: {targetset}")

    sourceset.class_to_idx = allowed_classes
    return sourceset

def AlignClassToIdx_TinyImageNet(sourceset, targetset_class_to_idx):
    for class_identifier in sourceset.classes:
        if class_identifier in targetset_class_to_idx:
            sourceset.class_to_idx[class_identifier] = targetset_class_to_idx[class_identifier]
    return sourceset
 