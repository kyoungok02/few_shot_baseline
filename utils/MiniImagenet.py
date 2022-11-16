import numpy as np
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

root_dir = "/home/few_shot/few_shot_baseline"

class MiniImageNet(Dataset):
    def __init__(self, data_dir, split='train'):
        self.data_dir = data_dir
        self.split = split
        if split == 'train':
            self.image_dir = os.path.join(data_dir,"train")
        elif split == 'val':
            self.image_dir = os.path.join(data_dir,"val")
        elif split == 'test':
            self.image_dir = os.path.join(data_dir,"test")
        classes, class_to_idx = self.find_classes(self.image_dir)
        self.classes = classes
        self.class_to_idx = class_to_idx

        self.images, self.labels = self.make_dataset(self.image_dir, class_to_idx)
        assert (len(self.images) == len(self.labels))
        self.transform = transforms.ToTensor()

    # dataset의 class 이름, class index 저장
    def find_classes(self,dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    # txt 파일로 저장된 train, test data의 경로를 읽어와서 image와 label을 페어로 저장
    def make_dataset(self, datadir, class_to_idx):
        images = []
        labels = []
        image_list = os.listdir(datadir)
        for label in image_list:
            image_dir = os.path.join(datadir,label)
            for file_name in os.listdir(image_dir):
                image_path = os.path.join(image_dir,file_name)
                _img = image_path
                assert os.path.isfile(_img)
                images.append(_img)
                labels.append(class_to_idx[label])
        return images, labels

    def __getitem__(self,index):
        _img = Image.open(self.images[index]).convert('RGB')
        _label = self.labels[index]
        if self.transform is not None:	# image에 지정된 transform 수행(tensor 변환 포함)
            _img = self.transform(_img)
        return _img, _label

    def __len__(self):
        return len(self.images)


class MetaMiniImageNet(MiniImageNet):
    def __init__(self, data_dir, split='train', batchsz=8, n_way=5, k_shot=5, k_query=15, imsize=38):
        self.data_dir = data_dir
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        self.imsize = imsize
        self.batchsz = batchsz
        self.split = split
        self.support_size = self.n_way * self.k_shot
        self.query_size = self.n_way * self.k_query
        if split == 'train':
            self.image_dir = os.path.join(data_dir,"train")
        elif split == 'val':
            self.image_dir = os.path.join(data_dir,"val")
        elif split == 'test':
            self.image_dir = os.path.join(data_dir,"test")
        classes, class_to_idx = self.find_classes(self.image_dir)
        self.classes = classes
        self.class_to_idx = class_to_idx

        self.images, self.labels = self.make_dataset(self.image_dir, class_to_idx)
        assert (len(self.images) == len(self.labels))
        self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                                 transforms.Resize((self.imsize, self.imsize)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])       
        self.create_batch(self.batchsz)

    def create_batch(self, meta_iterations):
        """
        create batch for meta-learning.
        ×episode× here means batch, and it means how many sets we want to retain.
        :param episodes: batch size
        :return:
        """
        self.support_x_batch = []  # support set batch
        self.query_x_batch = []  # query set batch
        for b in range(meta_iterations):  # for each batch
            # 1.select n_way classes randomly
            selected_cls = np.random.choice(self.classes, self.n_way, False)  # no duplicate
            np.random.shuffle(selected_cls)
            support_x = []
            query_x = []
            for cls in selected_cls:
                # 2. select k_shot + k_query for each class
                image_path = os.path.join(self.image_dir,cls)
                selected_imgs_idx = np.random.choice(os.listdir(image_path), self.k_shot + self.k_query, False)
                indices_support = np.array(selected_imgs_idx[:self.k_shot])  # idx for Dtrain
                indices_query = np.array(selected_imgs_idx[self.k_shot:])  # idx for Dtest
                indices_support = [os.path.join(image_path,i) for i in indices_support]
                indices_query = [os.path.join(image_path,i) for i in indices_query]
                support_x.extend(indices_support)  # get all image filenames
                query_x.extend(indices_query)

            # shuffle the corresponding relation between support set and query set
            support_x = np.random.permutation(support_x)
            query_x = np.random.permutation(query_x)

            self.support_x_batch.append(support_x)  # append set to current sets
            self.query_x_batch.append(query_x)  # append sets to current sets

    def __getitem__(self, index):
        """
        index means index of sets, 0<= index <= batchsz-1
        :param index:
        :return:
        """

        # initialise empty tensors for the images
        support_x = torch.FloatTensor(self.support_size, 3, self.imsize, self.imsize)
        support_y = np.chararray(self.support_size,itemsize=9)
        query_x = torch.FloatTensor(self.query_size, 3, self.imsize, self.imsize)
        query_y = np.chararray(self.query_size,itemsize=9)

        # get the filenames and labels of the images
        filenames_support_x = [item for item in self.support_x_batch[index]]
        filenames_query_x = [item for item in self.query_x_batch[index]]
            
        for i, path in enumerate(filenames_support_x):
        # filename: n0153282900000005.jpg, first 9 chars are label
            support_x[i] = self.transform(path)
            support_y[i] = np.array(path.split('/')[-2])

        for i, path in enumerate(filenames_query_x):
            query_x[i] = self.transform(path)        
            query_y[i] = np.array(path.split('/')[-2])
        # unique: [n-way], sorted
        unique = np.random.permutation(np.unique(support_y))
        # relative means the label ranges from 0 to n-way
        support_y_relative = np.zeros(self.support_size)
        query_y_relative = np.zeros(self.query_size)
        for idx, l in enumerate(unique):
            support_y_relative[support_y == l] = idx
            query_y_relative[query_y == l] = idx

        return support_x, torch.LongTensor(support_y_relative), query_x, torch.LongTensor(query_y_relative)
    
    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return self.batchsz

class MetaMiniImageNet(MiniImageNet):
    def __init__(self, data_dir, split='train', batchsz=8, n_way=5, k_shot=5, k_query=15, imsize=38):
        self.data_dir = data_dir
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        self.imsize = imsize
        self.batchsz = batchsz
        self.split = split
        self.support_size = self.n_way * self.k_shot
        self.query_size = self.n_way * self.k_query
        if split == 'train':
            self.image_dir = os.path.join(data_dir,"train")
        elif split == 'val':
            self.image_dir = os.path.join(data_dir,"val")
        elif split == 'test':
            self.image_dir = os.path.join(data_dir,"test")
        classes, class_to_idx = self.find_classes(self.image_dir)
        self.classes = classes
        self.class_to_idx = class_to_idx

        self.images, self.labels = self.make_dataset(self.image_dir, class_to_idx)
        assert (len(self.images) == len(self.labels))
        self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                                 transforms.Resize((self.imsize, self.imsize)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])       
        self.create_batch(self.batchsz)

    def create_batch(self, meta_iterations):
        """
        create batch for meta-learning.
        ×episode× here means batch, and it means how many sets we want to retain.
        :param episodes: batch size
        :return:
        """
        self.support_x_batch = []  # support set batch
        self.query_x_batch = []  # query set batch
        for b in range(meta_iterations):  # for each batch
            # 1.select n_way classes randomly
            selected_cls = np.random.choice(self.classes, self.n_way, False)  # no duplicate
            np.random.shuffle(selected_cls)
            support_x = []
            query_x = []
            for cls in selected_cls:
                # 2. select k_shot + k_query for each class
                image_path = os.path.join(self.image_dir,cls)
                selected_imgs_idx = np.random.choice(os.listdir(image_path), self.k_shot + self.k_query, False)
                indices_support = np.array(selected_imgs_idx[:self.k_shot])  # idx for Dtrain
                indices_query = np.array(selected_imgs_idx[self.k_shot:])  # idx for Dtest
                indices_support = [os.path.join(image_path,i) for i in indices_support]
                indices_query = [os.path.join(image_path,i) for i in indices_query]
                support_x.extend(indices_support)  # get all image filenames
                query_x.extend(indices_query)

            # shuffle the corresponding relation between support set and query set
            support_x = np.random.permutation(support_x)
            query_x = np.random.permutation(query_x)

            self.support_x_batch.append(support_x)  # append set to current sets
            self.query_x_batch.append(query_x)  # append sets to current sets

    def __getitem__(self, index):
        """
        index means index of sets, 0<= index <= batchsz-1
        :param index:
        :return:
        """

        # initialise empty tensors for the images
        support_x = torch.FloatTensor(self.support_size, 3, self.imsize, self.imsize)
        support_y = np.chararray(self.support_size,itemsize=9)
        query_x = torch.FloatTensor(self.query_size, 3, self.imsize, self.imsize)
        query_y = np.chararray(self.query_size,itemsize=9)

        # get the filenames and labels of the images
        filenames_support_x = [item for item in self.support_x_batch[index]]
        filenames_query_x = [item for item in self.query_x_batch[index]]
            
        for i, path in enumerate(filenames_support_x):
        # filename: n0153282900000005.jpg, first 9 chars are label
            support_x[i] = self.transform(path)
            support_y[i] = np.array(path.split('/')[-2])

        for i, path in enumerate(filenames_query_x):
            query_x[i] = self.transform(path)        
            query_y[i] = np.array(path.split('/')[-2])
        # unique: [n-way], sorted
        unique = np.random.permutation(np.unique(support_y))
        # relative means the label ranges from 0 to n-way
        support_y_relative = np.zeros(self.support_size)
        query_y_relative = np.zeros(self.query_size)
        for idx, l in enumerate(unique):
            support_y_relative[support_y == l] = idx
            query_y_relative[query_y == l] = idx

        return support_x, torch.LongTensor(support_y_relative), query_x, torch.LongTensor(query_y_relative)
    
    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return self.batchsz

class SiamMiniImageNetTrain(MiniImageNet):
    def __init__(self, data_dir, batchsz=8, n_way=5, k_shot=5,imsize=38):
        self.data_dir = data_dir
        self.n_way = n_way
        self.k_shot = k_shot
        self.imsize = imsize
        self.batchsz = batchsz
        self.support_size = self.n_way * self.k_shot
        self.image_dir = os.path.join(data_dir,"train")
        classes, class_to_idx = self.find_classes(self.image_dir)
        self.classes = classes
        self.class_to_idx = class_to_idx

        self.images, self.labels = self.make_dataset(self.image_dir, class_to_idx)
        assert (len(self.images) == len(self.labels))
        self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                                 transforms.Resize((self.imsize, self.imsize)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])       
        self.create_batch(self.batchsz)

    def create_batch(self, meta_iterations):
        """
        create batch for meta-learning.
        ×episode× here means batch, and it means how many sets we want to retain.
        :param episodes: batch size
        :return:
        """
        self.x1_batch = []  # support set batch
        self.x2_batch = []  # query set batch
        self.labels_batch = []
        for b in range(meta_iterations):  # for each batch
            # 1.select n_way classes randomly
            selected_cls = np.random.choice(self.classes, self.n_way, False)  # no duplicate
            np.random.shuffle(selected_cls)
            x1 = []
            x2 = []
            labels = []
            for cls in selected_cls:
                # 2. select k_shot + k_query for each class
                image_path = os.path.join(self.image_dir,cls)
                selected_imgs_idx = np.random.choice(os.listdir(image_path), self.k_shot, False)
                induce_x1 = []
                induce_x2 = []
                idc_label = []
                for i, img in enumerate(selected_imgs_idx):
                    # get image from same class
                    if i %2 == 1:
                        image1 = img
                        image2 = np.random.choice(os.listdir(image_path))
                        label = 1.0
                    else :
                        other_cls = np.random.choice(self.classes)
                        while cls == other_cls:
                            other_cls = np.random.choice(self.classes)
                        image1 = img
                        image2 = np.random.choice(os.listdir(os.path.join(self.image_dir,other_cls)))
                        label = 0.0
                    induce_x1.append(image1)
                    induce_x2.append(image2)
                    idc_label.append(label)
                induce_x1 = [os.path.join(image_path,i) for i in induce_x1]
                induce_x2 = [os.path.join(image_path,i) for i in induce_x2]
                x1.extend(induce_x1)  # get all image filenames
                x2.extend(induce_x1)
                labels.extend(idc_label)

            self.x1_batch.append(x1)  # append set to current sets
            self.x2_batch.append(x2)  # append sets to current sets
            self.labels_batch.append(labels)

    def __getitem__(self, index):
        """
        index means index of sets, 0<= index <= batchsz-1
        :param index:
        :return:
        """

        # initialise empty tensors for the images
        x1 = torch.FloatTensor(self.support_size, 3, self.imsize, self.imsize)
        x2 = torch.FloatTensor(self.support_size, 3, self.imsize, self.imsize)
        y = torch.FloatTensor(self.support_size)

        # get the filenames and labels of the images
        filenames_x1 = [item for item in self.x1_batch[index]]
        filenames_x2 = [item for item in self.x2_batch[index]]
        label_y = [item for item in self.labels_batch[index]]
            
        for i, path in enumerate(filenames_x1):
            x1[i] = self.transform(path)
            x2[i] = self.transform(filenames_x2[i])
            y[i] = label_y[i]
        return x1, x2, y
    
    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return self.batchsz

class SiamMiniImageNetTest(MiniImageNet):
    def __init__(self, data_dir, split = "val", batchsz=8, n_way=5, k_query=15, imsize=38):
        self.data_dir = data_dir
        self.n_way = n_way
        self.k_query = k_query
        self.imsize = imsize
        self.batchsz = batchsz
        self.query_size = self.n_way * self.k_query
        if split == 'val':
            self.image_dir = os.path.join(data_dir,"val")
        elif split == 'test':
            self.image_dir = os.path.join(data_dir,"test")
        classes, class_to_idx = self.find_classes(self.image_dir)
        self.classes = classes
        self.class_to_idx = class_to_idx

        self.images, self.labels = self.make_dataset(self.image_dir, class_to_idx)
        assert (len(self.images) == len(self.labels))
        self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                                 transforms.Resize((self.imsize, self.imsize)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])       
        self.create_batch(self.batchsz)

    def create_batch(self, meta_iterations):
        """
        create batch for meta-learning.
        ×episode× here means batch, and it means how many sets we want to retain.
        :param episodes: batch size
        :return:
        """
        self.x1_batch = []  # support set batch
        self.x2_batch = []  # query set batch
        self.labels_batch = []
        for b in range(meta_iterations):  # for each batch
            # 1.select n_way classes randomly
            selected_cls = np.random.choice(self.classes, self.n_way, False)  # no duplicate
            np.random.shuffle(selected_cls)
            x1 = []
            x2 = []
            labels = []
            for cls in selected_cls:
                # 2. select k_shot + k_query for each class
                image_path = os.path.join(self.image_dir,cls)
                selected_imgs_idx = np.random.choice(os.listdir(image_path), self.k_query, False)
                induce_x1 = []
                induce_x2 = []
                idc_label = []
                for i, img in enumerate(selected_imgs_idx):
                    # get image from same class
                    if i == 0:
                        image1 = img
                        image2 = np.random.choice(os.listdir(image_path))
                        label = 1.0
                    else :
                        other_cls = np.random.choice(self.classes)
                        while cls == other_cls:
                            other_cls = np.random.choice(self.classes)
                        image1 = img
                        image2 = np.random.choice(os.listdir(os.path.join(self.image_dir,other_cls)))
                        label = 0.0
                    induce_x1.append(image1)
                    induce_x2.append(image2)
                    idc_label.append(label)
                induce_x1 = [os.path.join(image_path,i) for i in induce_x1]
                induce_x2 = [os.path.join(image_path,i) for i in induce_x2]
                x1.extend(induce_x1)  # get all image filenames
                x2.extend(induce_x1)
                labels.extend(idc_label)
            self.x1_batch.append(x1)  # append set to current sets
            self.x2_batch.append(x2)  # append sets to current sets
            self.labels_batch.append(labels)

    def __getitem__(self, index):
        """
        index means index of sets, 0<= index <= batchsz-1
        :param index:
        :return:
        """

        # initialise empty tensors for the images
        x1 = torch.FloatTensor(self.query_size, 3, self.imsize, self.imsize)
        x2 = torch.FloatTensor(self.query_size, 3, self.imsize, self.imsize)
        y = torch.FloatTensor(self.query_size)

        # get the filenames and labels of the images
        filenames_x1 = [item for item in self.x1_batch[index]]
        filenames_x2 = [item for item in self.x2_batch[index]]
        label_y = [item for item in self.labels_batch[index]]
            
        for i, path in enumerate(filenames_x1):
            x1[i] = self.transform(path)
            x2[i] = self.transform(filenames_x2[i])
            y[i] = label_y[i]
        return x1, x2, y
    
    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return self.batchsz

if __name__ == '__main__':
    # the following episode is to view one set of images via tensorboard.
    from torchvision.utils import make_grid
    from matplotlib import pyplot as plt
    from tensorboardX import SummaryWriter
    import time

    plt.ion()

    tb = SummaryWriter('runs', 'mini-imagenet')
    imagnet_dir = os.path.join(root_dir,"data/mini-imagenet")
    train_dataset = MetaMiniImageNet(imagnet_dir,split="train",batchsz=1000,n_way=5, k_shot=5,k_query=15, imsize=168)
    # dl_train = DataLoader(train_dataset,batch_size=4)
    for i, set_ in enumerate(train_dataset):
        # support_x: [k_shot*n_way, 3, 84, 84]
        support_x, support_y, query_x, query_y = set_

        support_x = make_grid(support_x, nrow=2)
        query_x = make_grid(query_x, nrow=2)

        plt.figure(1)
        plt.imshow(support_x.transpose(2, 0).numpy())
        plt.pause(0.5)
        plt.figure(2)
        plt.imshow(query_x.transpose(2, 0).numpy())
        plt.pause(0.5)

        tb.add_image('support_x', support_x)
        tb.add_image('query_x', query_x)

        time.sleep(5)

    tb.close() 
    
