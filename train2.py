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


tf.random.set_seed(1234)


DATA_DIR = '/kaggle/input/right20230327-50/right20230327_50/train'
print('[debug] - DATA_DIR', DATA_DIR)

config = {
    'train_ds' : '/kaggle/working/pointnet3/data/modelnet_train.tfrecord',
    'val_ds' : '/kaggle/working/pointnet3/data/modelnet_val.tfrecord',
    'batch_size' : 16,
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

#################################################################################
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

from pnet2_layers.layers import Pointnet_SA, Pointnet_SA_MSG

#############################################
# self._set_inputs(tf.TensorSpec([batch_size, 1071, 3], tf.float32, name='inputs'))
#############################################

class CLS_MSG_Model(Model):
    def __init__(self, batch_size, num_classes, bn=False, activation=tf.nn.relu):
        super(CLS_MSG_Model, self).__init__()
        print('[debug-CLS_MSG_Model-init] -----------------------------')
        self.activation = activation
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.bn = bn
        self.keep_prob = 0.4
        self.kernel_initializer = 'glorot_normal'
        self.kernel_regularizer = None
        self.init_network()

    def init_network(self):
        print('[debug-CLS_MSG_Model-init_network] -----------------------------')
        self.layer1 = Pointnet_SA_MSG(
            npoint=1024,
            radius_list=[0.1,0.2,0.4],
            nsample_list=[16,32,128],
            mlp=[[32,32,64], [64,64,128], [64,96,128]],
            activation=self.activation,
            bn = self.bn
        )

        self.layer2 = Pointnet_SA_MSG(
            npoint=512,
            radius_list=[0.2,0.4,0.8],
            nsample_list=[32,64,128],
            mlp=[[64,64,128], [128,128,256], [128,128,256]],
            activation=self.activation,
            bn = self.bn
        )

        self.layer3 = Pointnet_SA(
            npoint=None,
            radius=None,
            nsample=None,
            mlp=[256, 512, 1024],
            group_all=True,
            activation=self.activation,
            bn = self.bn
        )

        self.dense1 = Dense(512, activation=self.activation)
        self.dropout1 = Dropout(self.keep_prob)
        self.dense2 = Dense(128, activation=self.activation)
        self.dropout2 = Dropout(self.keep_prob)
        self.dense3 = Dense(self.num_classes, activation=tf.nn.softmax)
    
    def forward_pass(self, input, training):
        print('[debug-CLS_MSG_Model-forward_pass] -----------------------------')
        xyz, points = self.layer1(input, None, training=training)
        xyz, points = self.layer2(xyz, points, training=training)
        xyz, points = self.layer3(xyz, points, training=training)
        net = tf.reshape(points, (self.batch_size, -1))
        net = self.dense1(net)
        net = self.dropout1(net)
        net = self.dense2(net)
        net = self.dropout2(net)
        pred = self.dense3(net)
        return pred

    def train_step(self, input):
        print('[debug-CLS_MSG_Model-train_step] -----------------------------')
        with tf.GradientTape() as tape:
            pred = self.forward_pass(input[0], True)
            loss = self.compiled_loss(input[1], pred)
		
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(input[1], pred)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, input):
        print('[debug-CLS_MSG_Model-test_step] -----------------------------')
        pred = self.forward_pass(input[0], False)
        loss = self.compiled_loss(input[1], pred)
        self.compiled_metrics.update_state(input[1], pred)
        return {m.name: m.result() for m in self.metrics}

    def call(self, input, training=False):
        print('[debug-CLS_MSG_Model-call] -----------------------------')
        return self.forward_pass(input, training)


#################################################################################
train_ds = train_dataset
val_ds = valid_dataset
test_ds = test_dataset
print('[debug] - ------------------dataset detail--------------------')
print('[debug] - >>train_ds', train_ds)
print('[debug] - >>val_ds', val_ds)
print('[debug] - >>test_ds', test_ds)


model = CLS_MSG_Model(config['batch_size'], NUM_CLASSES, config['bn'])

early_stop = EarlyStopping(
    monitor='sparse_categorical_accuracy', min_delta=0, patience=5, verbose=0, mode='auto',
    baseline=None, restore_best_weights=False
)
learning_rate_reduction = ReduceLROnPlateau(monitor='sparse_categorical_accuracy', 
                                        patience=3, 
                                        verbose=1, 
                                        factor=0.5, 
                                        min_lr=0.00001)
checkpoint_path = "./logs/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=2, save_weights_only=True, save_freq='epoch')
#model.save_weights(checkpoint_path.format(epoch=0))

print('[debug] - ------------------build before--------------------')
model.build((config['batch_size'], NUM_POINTS, 3))
print('[debug] - ------------------build after--------------------')
print(model.summary())

model.compile(
    optimizer=keras.optimizers.Adam(config['lr']),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()]
)

history = model.fit(
    train_ds,
    validation_data = val_ds,
    callbacks=[learning_rate_reduction, early_stop, cp_callback],
    epochs = config['epochs'],
    verbose = True
)
#############################################################
print('>>>>>>>>>>>>>>>>>>>>>>>>save model and do convertion<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
#tf.saved_model.save(model, '/kaggle/working/model')
model.save_weights('model_sign.model', overwrite=True, save_format='tf')

'''
#转tflite
#https://blog.csdn.net/bjbz_cxy/article/details/120503631
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
tflite_model = converter.convert()
 
open("./model_binary.tflite","wb").write(tflite_model)
'''
#############################################################
print('>>>>>>>>>>>>>>>>>>>>>>>>history<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
print(history.history)



##############################################################
print('>>>>>>>>>>>>>>>>>>>>>>>>evaluate<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

test_loss, test_acc = model.evaluate(test_dataset, verbose=2)
print('[debug] - Test loss:', test_loss)
print('[debug] - Test accuracy:', test_acc)

'''
print('>>>>>>>>>>>>>>>>>>>>>>>>predict 1<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

data = test_dataset.take(1)
print('data:', test_dataset)

points, labels = list(data)[0]

print('points shape:', points.shape)
print('labels shape:', labels.shape)
print('labels:', labels)


# run test data through model
preds = model.predict(points)
#print('>>preds:', preds)
predicts = tf.math.argmax(preds, -1)
print('predicts:', predicts)




print('>>>>>>>>>>>>>>>>>>>>>>>>predict 2<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

def get_batch_model(model, batch_size=1):
    n_batch = batch_size
    print('[debug] - model batch size ', n_batch)
    model2 = PointConvModel(n_batch, config['bn'], num_classes=NUM_CLASSES)
    weights = model.get_weights()
    model2.build((n_batch, NUM_POINTS, 3))

    model2.set_weights(weights)
    return model2
    
example_images1 = []
example_labels1 = []

#获取10个batch的数据
for batch in test_dataset:
    example_images1.append(batch[0].numpy())
    example_labels1.append(batch[1].numpy())

example_images1 = np.concatenate(example_images1, axis = 0)    
example_labels1 = np.concatenate(example_labels1, axis = 0)   
print('[debug] - >>example_images1 shape', example_images1.shape)
print('[debug] - >>example_labels1 shape', example_labels1.shape)
print('[debug] - >>example_images1 shape[0]', example_images1.shape[0])

print('[debug] - >>example_images1[0] shape', example_images1[0].shape)
print('[debug] - >>example_labels1[0] shape', example_labels1[0].shape)



print('[debug] - >>start single predict------------------------------------------')

example_images = []
example_labels = []

#获取10个batch的数据
for batch in test_dataset:
    example_images.append([batch[0].numpy()[0]])
    example_labels.append([batch[1].numpy()[0]])
    break

example_images = np.concatenate(example_images, axis = 0)    
example_labels = np.concatenate(example_labels, axis = 0)    

print('[debug] - >>example_images shape', example_images.shape)
print('[debug] - >>example_labels shape', example_labels.shape)
print('[debug] - >>example_images shape[0]', example_images.shape[0])

print('[debug] - >>example_images[0] shape', example_images[0].shape)
print('[debug] - >>example_labels[0] shape', example_labels[0].shape)

print('[debug] - >>example_labels', example_labels)

n_batch = example_images.shape[0]
model2 = get_batch_model(model, n_batch)
Y_pred = model2.predict(example_images, batch_size = n_batch)
print('[debug] - >>n_batch', n_batch)
with printoptions(precision=6, suppress=True):
    print('[debug] - >>Y_pred', Y_pred)
Y_pred = tf.math.argmax(Y_pred, -1)
print('[debug] - predicts:', Y_pred)
'''

