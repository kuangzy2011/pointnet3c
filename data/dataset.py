import os
import sys
import glob
import trimesh
import random
import numpy as np

class DataSetHand():
    def __init__(self, datapath, batch_size, augment = 3):
        self.data_path = datapath
        self.batch_size = batch_sie
        self.augment = augment
        self.num_points = 0
        self.class_maps = []
      
    def load_label(self):
        label_file = os.path.join(self.data_path, "label.txt")
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

    def split_train(self, full_list, shuffle=False,ratio=0.2, ratio2 = 0.5):
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
    
    def parse_dataset():
        train_points = []
        train_labels = []
        test_points = []
        test_labels = []
        valid_points = []
        valid_labels = []
        class_map = self.load_label()
        num_class = len(class_map)
        
        file_cnt = []
               
        #get files number of each category， Then take the mininum number as training set to balanced training set 
        for key in class_map.keys():
            key_value = self.generate_key(num_class, key)
            # store folder name with ID so we can retrieve later
            #class_map[i] = folder.split("/")[-1]
            
            folder = os.path.join(self.data_path, class_map[key])
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
            
            folder = os.path.join(self.data_path, class_map[key])
            # gather all files
            all_files = glob.glob(os.path.join(folder, "*.off"))
            
            for i in range(100):
                train_files1, all_files, valid_files1 = self.split_train(all_files, True, ratio=1, ratio2=0)
    
            
            train_files, test_files, valid_files = self.split_train(all_files[:min_files], True, ratio=0.1, ratio2=0.5)
    
            #loop = 0
            for f in train_files:
                #loop += 1
                #print('>>>load ', loop, f)
                points = trimesh.load(f).vertices
                if self.num_points == 0:
                    self.num_points = points.shape[0]
                train_points.append(points)
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

    def dataset(self):
        train_points, test_points, valid_points, train_labels, test_labels, valid_labels, self.class_maps = parse_dataset()
        train_points, train_labels = data_augment(train_points, train_labels, self.argument)
      
        self.train_dataset = tf.data.Dataset.from_tensor_slices((train_points, train_labels))
        self.test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels))
        self.valid_dataset = tf.data.Dataset.from_tensor_slices((valid_points, valid_labels))

        self.train_dataset = train_dataset.shuffle(len(train_points)).map(augment).batch(self.batch_size, drop_remainder=True)
        self.test_dataset = test_dataset.shuffle(len(test_points)).batch(self.batch_size, drop_remainder=True)
        self.valid_dataset = valid_dataset.shuffle(len(valid_points)).batch(self.batch_size, drop_remainder=True)

        return self.train_dataset, self.test_dataset, self.valid_dataset, self.class_maps

        
