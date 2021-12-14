import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

#################################### GLOBAL VARIABLES ####################################

CLASS_LIST = ['void',
              'vegetation', 
              'pedestrian', 
              'sky', 
              'road', 
              'sidewalk', 
              'construction',
              'transportation',
              'sign',
              'animal',
              'other',
              'poleLike']
              
N_CLASSES = len(CLASS_LIST)
TARGET_SIZE = (2048, 1024)
COLORS = [ (0, 0, 0),        # void
           (20, 250, 35),    # vegetation
           (220, 20, 60),    # pedestrian
           (70, 180, 180),   # sky
           (128, 64, 128),   # road
           (244, 35, 232),   # sidewalk
           (70, 70, 70),     # construction
           (0, 20, 142),     # transportation
           (220, 220, 0),    # sign
           (100, 100, 10),   # animal
           (111, 74, 0),     # other
           (153, 153, 153)]  # poleLike


DATASET_SAMPLES = 0
TRAIN_SAMPLES = 0
TEST_SAMPLES = 0
VAL_SAMPLES = 0
DS_CLASS_PRECENTAGE = [0] * N_CLASSES
N = 1000

# clearing '._*.png' files
try:
    for file in os.listdir('camvid/images'):
        if file[0] == '.':
            os.remove(os.path.join('camvid/images', file))
except:
    pass


SAVE_DIR = 'data'
if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)
splits = ['test', 'val', 'train']


csv_file_name = 'dataset_labels.csv'
csv_data = { 'name': CLASS_LIST,
             'r': [ str(x[0]) for x in COLORS],
             'g': [ str(x[1]) for x in COLORS],
             'b': [ str(x[2]) for x in COLORS]}

df = pd.DataFrame( csv_data, columns=['name', 'r', 'g', 'b'])
df.to_csv(os.path.join(SAVE_DIR, csv_file_name), index=False)

camvid_class_list= ['Animal',
                    'Archway', 
                    'Bicyclist', 
                    'Bridge', 
                    'Building', 
                    'Car', 
                    'CartLuggagePram', 
                    'Child', 
                    'Column_Pole', 
                    'Fence', 
                    'LaneMkgsDriv', 
                    'LaneMkgsNonDriv', 
                    'Misc_Text', 
                    'MotorcycleScooter', 
                    'OtherMoving', 
                    'ParkingBlock', 
                    'Pedestrian', 
                    'Road', 
                    'RoadShoulder', 
                    'Sidewalk', 
                    'SignSymbol', 
                    'Sky', 
                    'SUVPickupTruck', 
                    'TrafficCone', 
                    'TrafficLight', 
                    'Train', 
                    'Tree', 
                    'Truck_Bus', 
                    'Tunnel', 
                    'VegetationMisc', 
                    'Void', 
                    'Wall']

camvid_class_transition = {
                    'Animal': 'animal',
                    'Archway': 'construction', 
                    'Bicyclist': 'transportation', 
                    'Bridge': 'construction', 
                    'Building': 'construction', 
                    'Car': 'transportation', 
                    'CartLuggagePram': 'other', 
                    'Child': 'pedestrian', 
                    'Column_Pole': 'poleLike', 
                    'Fence': 'construction', 
                    'LaneMkgsDriv': 'road', 
                    'LaneMkgsNonDriv': 'road', 
                    'Misc_Text': 'sign', 
                    'MotorcycleScooter': 'transportation', 
                    'OtherMoving': 'transportation', 
                    'ParkingBlock': 'construction', 
                    'Pedestrian': 'pedestrian', 
                    'Road': 'road', 
                    'RoadShoulder': 'road', 
                    'Sidewalk': 'sidewalk', 
                    'SignSymbol': 'sign', 
                    'Sky': 'sky', 
                    'SUVPickupTruck': 'transportation', 
                    'TrafficCone': 'other', 
                    'TrafficLight': 'poleLike', 
                    'Train': 'transportation', 
                    'Tree': 'vegetation', 
                    'Truck_Bus': 'transportation', 
                    'Tunnel': 'construction', 
                    'VegetationMisc': 'vegetation', 
                    'Void': 'void',
                    'Wall': 'construction'}

cityscapes_class_list = ['unlabeled',
                         'ego vehicle',
                         'rectification border',
                         'out of roi',
                         'static',
                         'dynamic',
                         'ground',
                         'road',
                         'sidewalk',
                         'parking',
                         'rail track',
                         'building',
                         'wall',
                         'fence',
                         'guard rail',
                         'bridge',
                         'tunnel',
                         'pole',
                         'polegroup',
                         'traffic light',
                         'traffic sign',
                         'vegetation',
                         'terrain',
                         'sky',
                         'person',
                         'rider',
                         'car',
                         'truck',
                         'bus',
                         'caravan',
                         'trailer',
                         'train',
                         'motorcycle',
                         'bicycle']

cityscapes_class_transition = { 'unlabeled': 'void',
                                'ego vehicle': 'void',
                                'rectification border': 'void',
                                'out of roi': 'void',
                                'static': 'other',
                                'dynamic': 'other',
                                'ground': 'sidewalk',
                                'road': 'road',
                                'sidewalk': 'sidewalk',
                                'parking': 'road',
                                'rail track': 'other',
                                'building': 'construction',
                                'wall': 'construction',
                                'fence': 'construction',
                                'guard rail': 'construction',
                                'bridge': 'construction',
                                'tunnel': 'construction',
                                'pole': 'poleLike',
                                'polegroup': 'poleLike',
                                'traffic light': 'other',
                                'traffic sign': 'sign',
                                'vegetation': 'vegetation',
                                'terrain': 'vegetation',
                                'sky': 'sky',
                                'person': 'pedestrian',
                                'rider': 'transportation',
                                'car': 'transportation',
                                'truck': 'transportation',
                                'bus': 'transportation',
                                'caravan': 'transportation',
                                'trailer': 'transportation',
                                'train': 'transportation',
                                'motorcycle': 'transportation',
                                'bicycle': 'transportation'}

#################################### FUNCTIONS ####################################

def lbl_color(lbl):
    """ Change label colors to new dataset colors."""
    idxs = np.unique(np.array(lbl))
    colored_lbl = np.zeros((3, lbl.size[1], lbl.size[0]), dtype=np.uint8 )
    for idx in idxs:
        r,g,b = COLORS[idx]
        mask = np.array(lbl) == idx
        colored_lbl[0][mask] = r
        colored_lbl[1][mask] = g
        colored_lbl[2][mask] = b
    
    colored_lbl = Image.fromarray(colored_lbl.transpose((1, 2, 0)), 'RGB')
    return colored_lbl

def id_transition(old_idx, dataset):
    """ Id transition to new dataset."""
    if dataset == 'camvid':
        class_list = camvid_class_list
        class_transition = camvid_class_transition
    elif dataset == 'cityscapes':
        class_list = cityscapes_class_list
        class_transition = cityscapes_class_transition
    else:
        print('No {} dataset defined.'.format(dataset))

    class_name = class_list[old_idx]
    new_class_name = class_transition[class_name]
    new_class_idx = CLASS_LIST.index(new_class_name)
    return new_class_idx, new_class_name

def class_precentage(lbl):
    """ Calculate class percentage by its pixel index through dataset. """
    idxs = np.unique(np.array(lbl))
    img_size = TARGET_SIZE[0] * TARGET_SIZE[1]
    for idx in idxs:
        temp = np.zeros((lbl.size[1], lbl.size[0]), dtype=np.uint8 )
        temp[ np.array(lbl) == idx ] = 1
        DS_CLASS_PRECENTAGE[idx] += ( temp.sum() / img_size )

def camvid():
    """ process on CamVid Dataset. Change image size similar to Cityscapes dataset."""
    img_dir =  'camvid/images'
    lbl_dir = 'camvid/labels'

    try:
        os.remove(os.path.join(img_dir, '.DS_Store'))
    except:
        pass

    NoF = len(os.listdir(img_dir))
    NoF_test = int(NoF * 0.2)
    NoF_val = int(NoF * 0.2)
    NoF_train = NoF - NoF_test - NoF_val

    global DATASET_SAMPLES
    global TRAIN_SAMPLES
    global TEST_SAMPLES
    global VAL_SAMPLES

    cntr = 0
    cntr_train = 0
    cntr_val = 0
    cntr_test = 0
    for file_name in tqdm(sorted(os.listdir(img_dir)), desc='Camvid'):
        img = Image.open(os.path.join(img_dir, file_name)).convert("RGB")
        lbl = Image.open(os.path.join(lbl_dir, file_name).replace('.png', '_P.png')).convert("L")

        # changing idx to new class idx
        idxs = np.unique(np.asarray(lbl))
        new_lbl = np.zeros((lbl.size[1], lbl.size[0]), dtype=np.uint8 )
        for idx in idxs:
            new_idx, new_class = id_transition(idx, 'camvid')
            new_lbl[np.array(lbl) == idx] =  new_idx
        new_lbl = Image.fromarray(new_lbl , 'L') 
        
        resized_img = img.resize(TARGET_SIZE, Image.NEAREST)
        resized_lbl = new_lbl.resize(TARGET_SIZE, Image.NEAREST) 

        if cntr % 3 == 0 and cntr_test < NoF_test:
            saving_path_img = os.path.join(SAVE_DIR, splits[0], 'images', file_name)
            phase = 'test'
            cntr_test += 1
        elif cntr % 3 == 1 and cntr_val < NoF_val:
            saving_path_img = os.path.join(SAVE_DIR, splits[1], 'images', file_name)
            phase = 'val'
            cntr_val += 1
        else:
            saving_path_img = os.path.join(SAVE_DIR, splits[2], 'images', file_name)
            phase = 'train'
            cntr_train += 1
        saving_path_lbl = saving_path_img.replace('images', 'labels').replace('.png', '_labelIds.png')

        # Change color and calculate percentage
        colored_lbl = lbl_color(resized_lbl)
        class_precentage(resized_lbl)
        
        resized_img.save(saving_path_img)
        resized_lbl.save(saving_path_lbl)
        colored_lbl.save(saving_path_lbl.replace('_labelIds.png', '_color.png'))
        cntr += 1

    DATASET_SAMPLES += cntr
    TRAIN_SAMPLES += cntr_train
    TEST_SAMPLES += cntr_test
    VAL_SAMPLES += cntr_val
    print()
       
def cityscapes():
    """ process on Cityscapes dataset."""

    img_dir = 'leftImg8bit_trainvaltest/leftImg8bit'
    lbl_dir = 'gtFine_trainvaltest/gtFine'

    global DATASET_SAMPLES
    global TRAIN_SAMPLES
    global TEST_SAMPLES
    global VAL_SAMPLES

    phases = ['train', 'val']

    for phase in phases:
        cities_path = os.path.join(img_dir, phase)
        try:
            for file in os.listdir(cities_path):
                if file[0] == '.':
                    os.remove(os.path.join(cities_path, file))
        except:
            pass

        try:
            os.remove(os.path.join(cities_path, '.DS_Store'))
        except:
            pass

        for city in sorted(os.listdir(cities_path)):
            files_path = os.path.join(cities_path, city)
            try:
                for file in os.listdir(files_path):
                    if file[0] == '.':
                        print('Will be removed: ', os.path.join(files_path, file))
                        os.remove(os.path.join(files_path, file))
            except:
                pass

            NoF = len(os.listdir(files_path))
            NoF_test = int(NoF * 0.2)
            NoF_val = int(NoF * 0.2)
            NoF_train = NoF - NoF_test - NoF_val

            cntr = 0
            cntr_train = 0
            cntr_val = 0
            cntr_test = 0

            try:
                os.remove(os.path.join(files_path, '.DS_Store'))
            except:
                pass
            for file_name in tqdm(sorted(os.listdir(files_path)), desc='Cityscapes: ' + phase + '_' + city):
                img_path = os.path.join(files_path, file_name)
                lbl_path = img_path.replace(img_dir, lbl_dir).replace('_leftImg8bit.png', '_gtFine_labelIds.png')

                img = Image.open(img_path).convert('RGB')
                lbl = Image.open(lbl_path).convert('L')
 
                # changing idx to new class idx
                idxs = np.unique(np.asarray(lbl))
                new_lbl = np.zeros((lbl.size[1], lbl.size[0]), dtype=np.uint8 )
                for idx in idxs:
                    new_idx, new_class = id_transition(idx, 'cityscapes')
                    new_lbl[np.asarray(lbl) == idx] = new_idx
                lbl = Image.fromarray(new_lbl.astype(np.uint8), 'L')

                if cntr % 3 == 0 and cntr_test < NoF_test:
                    saving_path_img = os.path.join(SAVE_DIR, splits[0], 'images', file_name)
                    cntr_test += 1
                elif cntr % 3 == 1 and cntr_val < NoF_val:
                    saving_path_img = os.path.join(SAVE_DIR, splits[1], 'images', file_name)
                    cntr_val += 1
                else:
                    saving_path_img = os.path.join(SAVE_DIR, splits[2], 'images', file_name)
                    cntr_train += 1
                saving_path_lbl = saving_path_img.replace('images', 'labels').replace('.png', '_labelIds.png')
                
                # Change color and calculate percentage
                colored_lbl = lbl_color(lbl)
                class_precentage(lbl)
                
                img.save(saving_path_img)
                lbl.save(saving_path_lbl)
                colored_lbl.save(saving_path_lbl.replace('_labelIds.png', '_color.png'))
                cntr += 1
            
            DATASET_SAMPLES += cntr
            TRAIN_SAMPLES += cntr_train
            TEST_SAMPLES += cntr_test
            VAL_SAMPLES += cntr_val
    print()

def main():
    print('\n\t','------     *     ' * 5, '------\n\n' )

    camvid() 
    cityscapes()

    f = open(os.path.join(SAVE_DIR, 'about DATASET.txt'), "w+")
    str = 'Dataset class precentages:'
    f.writelines(str + '\n')
    print(str)
    
    for idx in range(N_CLASSES):
        str = '\t {} : {:.4f}'.format(CLASS_LIST[idx], DS_CLASS_PRECENTAGE[idx])
        f.writelines(str + '\n')
        print(str)
    str = '\t --  Total : {:.4f} (should be 1)'.format(np.array(DS_CLASS_PRECENTAGE).sum() / DATASET_SAMPLES )
    f.writelines(str + '\n')
    print(str)

    print()
    str = 'Dataset number of samples: {}, one sample size: {}, Number of Class: {}'.format(DATASET_SAMPLES, TARGET_SIZE, N_CLASSES)
    f.writelines('\n' + str)
    print(str)
    str = 'Splits: train({}) + val({}) + test({}) --- Total: {}'.format( TRAIN_SAMPLES, 
                                                                         VAL_SAMPLES, 
                                                                         TEST_SAMPLES, 
                                                                         TRAIN_SAMPLES + VAL_SAMPLES + TEST_SAMPLES)
    f.writelines('\n' + str)
    print(str)

    str = "\n'Custom Dataset' that combines 'CamVid' and 'Cityscapes' datasets."
    f.writelines('\n' + str)
    f.close()
    print()

###################################### RUN MAIN ######################################

if __name__ == '__main__':
    main()
    