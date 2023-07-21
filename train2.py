import os
import sys
import datetime
import os
import sys
import glob
import trimesh
import random
import numpy as np

sys.path.insert(0, './')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras

from models.cls_msg_model import CLS_MSG_Model
from models.cls_ssg_model import CLS_SSG_Model

tf.random.set_seed(1234)


DATA_DIR = '/kaggle/input/right20230327-50/right20230327_50/train'
print('[debug] - DATA_DIR', DATA_DIR)

config = {
    'train_ds' : '/kaggle/working/pointnet3/data/modelnet_train.tfrecord',
    'val_ds' : '/kaggle/working/pointnet3/data/modelnet_val.tfrecord',
    'batch_size' : 4,
    'lr' : 1e-3,
    'bn' : False,
    'log_dir' : 'modelnet_1',
    'epochs' : 1
}
print('[debug] - start train2 ==============================================================EPOCHS', config['epochs'])

def split_train(full_list,shuffle=False,ratio=0.2, ratio2 = 0.5):
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total==0 or offset<1:
        return [],[],full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    
    sub_total = len(sublist_1)
    offset = int(sub_total * ratio2)
    sublist_3 = sublist_1[:offset]
    sublist_4 = sublist_1[offset:]
    return sublist_2,sublist_4,sublist_3



#切分数据集也可以参考
#################################################################
#from sklearn.model_selection import train_test_split
#################################################################
def load_label():
    label_file = os.path.join(DATA_DIR, "label.txt")
    print('load label file {}'.format(label_file))
    d = {}
    with open(label_file) as f:
        d = eval(f.read())

    print('[debug] - label:', d)
    return d

def generate_key(l, k):
    #ret = np.zeros(l)
    #ret[k] = 1
    #return ret
    return k

#TODO, 这里用split_train随机切分训练集和测试集
def parse_dataset(num_points=21):

    train_points = []
    train_labels = []
    test_points = []
    test_labels = []
    valid_points = []
    valid_labels = []
    class_map = load_label()
    num_class = len(class_map)
    
    file_cnt = []
    
    
    #get files number of each category， Then take the mininum number as training set to balanced training set 
    for key in class_map.keys():
        key_value = generate_key(num_class, key)
        # store folder name with ID so we can retrieve later
        #class_map[i] = folder.split("/")[-1]
        
        folder = os.path.join(DATA_DIR, class_map[key])
        # gather all files
        all_files = glob.glob(os.path.join(folder, "*.off"))
        file_cnt.append(len(all_files))


    print('[debug] - file_cnt', file_cnt)
    print('[debug] - min', np.min(file_cnt))
    min_files = np.min(file_cnt)

    for key in class_map.keys():
        print("[debug] - processing class: {}".format(class_map[key]))
        key_value = generate_key(num_class, key)
        # store folder name with ID so we can retrieve later
        #class_map[i] = folder.split("/")[-1]
        
        folder = os.path.join(DATA_DIR, class_map[key])
        # gather all files
        all_files = glob.glob(os.path.join(folder, "*.off"))
        
        for i in range(100):
            train_files1, all_files, valid_files1 = split_train(all_files, True, ratio=1, ratio2=0)

        
        train_files, test_files, valid_files = split_train(all_files[:min_files], True, ratio=0.1, ratio2=0.5)

        #loop = 0
        for f in train_files:
            #loop += 1
            #print('>>>load ', loop, f)
            train_points.append(trimesh.load(f).vertices)
            train_labels.append(key_value)

        for f in test_files:
            test_points.append(trimesh.load(f).vertices)
            test_labels.append(key_value)

        for f in valid_files:
            valid_points.append(trimesh.load(f).vertices)
            valid_labels.append(key_value)

    return (
        np.array(train_points),
        np.array(test_points),
        np.array(valid_points),
        np.array(train_labels),
        np.array(test_labels),
        np.array(valid_labels),
        class_map,
    )


mesh_file = os.path.join(DATA_DIR, "Fist/Fist_1.jpg_227.off")
mesh = trimesh.load(mesh_file)

#TODO
points = mesh.vertices
print('[debug] - points shape', points.shape)


NUM_POINTS = points.shape[0]  #TODO
#NUM_CLASSES = 6  #TODO
BATCH_SIZE = config['batch_size']

train_points, test_points, valid_points, train_labels, test_labels, valid_labels, CLASS_MAP = parse_dataset(
    NUM_POINTS
)


print('[debug] - train_points, len', len(train_points))
print('[debug] - train_labels, len', len(train_labels))
print('[debug] - test_points, len', len(test_points))
print('[debug] - test_labels, len', len(test_labels))
print('[debug] - valid_points, len', len(valid_points))
print('[debug] - valid_labels, len', len(valid_labels))

NUM_CLASSES = len(CLASS_MAP)  #TODO

print('[debug] - CLASS MAP:', CLASS_MAP)


# 数据增强, 抖动增加数据
def data_augment(points, labels, zoomin=10):
    ret_points = points
    ret_labels = labels
    for i in range(zoomin):
        tmp_points = points
        if i % 5 == 0:
            tmp_points = points + tf.random.uniform(points.shape, -0.0001, 0.0001, dtype=tf.float64)
        elif i % 5 == 1:
            tmp_points = points + tf.random.uniform(points.shape, -0.0002, 0.0002, dtype=tf.float64)
        elif i % 5 == 2:
            tmp_points = points + tf.random.uniform(points.shape, -0.0003, 0.0003, dtype=tf.float64)
        elif i % 5 == 3:
            tmp_points = points + tf.random.uniform(points.shape, -0.0004, 0.0004, dtype=tf.float64)
        elif i % 5 == 4:
            tmp_points = points + tf.random.uniform(points.shape, -0.0005, 0.0005, dtype=tf.float64)
        else:
            tmp_points = points + tf.random.uniform(points.shape, -0.0005, 0.0005, dtype=tf.float64)
        ret_points = np.concatenate((ret_points, tmp_points))
        ret_labels = np.concatenate((ret_labels, labels))

    return ret_points, ret_labels

# 抖动 +　重新排序
def augment(points, label):
    # jitter points
    #points += tf.random.uniform(points.shape, -0.0005, 0.0005, dtype=tf.float64)
    # shuffle points, 按行打乱
    points = tf.random.shuffle(points)
    return points, label



print('[debug] - train_points, len', len(train_points))
print('[debug] - train_labels, len', len(train_labels), train_labels.shape)
print('[debug] - test_points, len', len(test_points))
print('[debug] - test_labels, len', len(test_labels))
print('[debug] - valid_points, len', len(valid_points))
print('[debug] - valid_labels, len', len(valid_labels))

train_points, train_labels = data_augment(train_points, train_labels, 10)
print('[debug] - After data augment...')
print('[debug] - train_points, len', len(train_points))
print('[debug] - train_labels, len', len(train_labels))

NUM_EXAMPLES = len(train_points)

#是把元组、列表和张量等数据进行特征切
train_dataset = tf.data.Dataset.from_tensor_slices((train_points, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels))
valid_dataset = tf.data.Dataset.from_tensor_slices((valid_points, valid_labels))

train_dataset = train_dataset.shuffle(len(train_points)).map(augment).batch(BATCH_SIZE, drop_remainder=True)
test_dataset = test_dataset.shuffle(len(test_points)).batch(BATCH_SIZE, drop_remainder=True)
valid_dataset = valid_dataset.shuffle(len(valid_points)).batch(BATCH_SIZE, drop_remainder=True)

print('[debug] - train_dataset', train_dataset)
print('[debug] - test_dataset', test_dataset)
print('[debug] - valid_dataset', valid_dataset)

print('==============================================')
print('[debug] - >>NUM_EXAMPLES', NUM_EXAMPLES)
print('[debug] - >>NUM_CLASSES', NUM_CLASSES)
print('[debug] - >>BATCH_SIZE', BATCH_SIZE)
print('[debug] - >>NUM_POINTS', NUM_POINTS)



model = CLS_MSG_Model(config['batch_size'], NUM_CLASSES, config['bn'])

callbacks = [
	keras.callbacks.EarlyStopping('val_sparse_categorical_accuracy', min_delta=0.01, patience=10),
	keras.callbacks.TensorBoard('./logs/{}'.format(config['log_dir']), update_freq=50),
	keras.callbacks.ModelCheckpoint('./logs/{}/model/weights.ckpt'.format(config['log_dir']), 'val_sparse_categorical_accuracy', save_best_only=True)
]

model.build(input_shape=(config['batch_size'], NUM_POINTS, 3))
print(model.summary())

model.compile(
	optimizer=keras.optimizers.Adam(config['lr']),
	loss=keras.losses.SparseCategoricalCrossentropy(),
	metrics=[keras.metrics.SparseCategoricalAccuracy()]
)

history = model.fit(
    train_dataset,
    validation_data = valid_dataset,
    callbacks = callbacks,
    epochs = config['epochs'],
    verbose = True
)

#############################################################
print('>>>>>>>>>>>>>>>>>>>>>>>>history<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
print(history.history)



##############################################################
print('>>>>>>>>>>>>>>>>>>>>>>>>evaluate<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

test_loss, test_acc = model.evaluate(test_dataset, verbose=2)
print('[debug] - Test loss:', test_loss)
print('[debug] - Test accuracy:', test_acc)

