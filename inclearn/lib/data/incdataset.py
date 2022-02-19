import logging
import random
import glob
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from .datasets import (
    APY, CUB200, LAD, AwA2, ImageNet100, ImageNet100UCIR, ImageNet1000, TinyImageNet200, iCIFAR10,
    iCIFAR100
)

logger = logging.getLogger(__name__)


class IncrementalDataset:

    def __init__(
        self,
        dataset_name,
        random_order=False,
        shuffle=True,
        workers=10,
        batch_size=128,
        seed=1,
        increment=10,
        validation_split=0.,
        onehot=False,
        initial_increment=None,
        sampler=None,
        sampler_config=None,
        data_path="data",
        class_order=None,
        dataset_transforms=None,
        all_test_classes=False,
        metadata_path=None
    ):
        datasets = _get_datasets(dataset_name)
        if metadata_path:
            print("Adding metadata path {}".format(metadata_path))
            datasets[0].metadata_path = metadata_path

        self._setup_data(
            datasets,
            random_order=random_order,
            class_order=class_order,
            seed=seed,
            increment=increment,
            validation_split=validation_split,
            initial_increment=initial_increment,
            data_path=data_path
        )

        dataset = datasets[0]()
        dataset.set_custom_transforms(dataset_transforms)
        self.train_transforms = dataset.train_transforms  # FIXME handle multiple datasets
        self.test_transforms = dataset.test_transforms
        self.common_transforms = dataset.common_transforms

        self.open_image = datasets[0].open_image

        self._current_task = 0

        self._seed = seed
        self._batch_size = batch_size
        self._workers = workers
        self._shuffle = shuffle
        self._onehot = onehot
        self._sampler = sampler
        self._sampler_config = sampler_config
        self._all_test_classes = all_test_classes

    @property
    def n_tasks(self):
        return len(self.increments)

    @property
    def n_classes(self):
        return sum(self.increments)

    def new_task_for_searching(self, memory=None, memory_val=None, memrate=None, task_id=None, disable_rmm=None):
        if self._current_task >= len(self.increments):
            raise Exception("No more tasks.")

        logger.info("Creating balanced subset")

        min_class = sum(self.increments[:self._current_task])
        max_class = sum(self.increments[:self._current_task + 1])
        
        if task_id != 0 and not disable_rmm:
            x_train, y_train = self._select_train(
                self.data_train, self.targets_train, low_range=min_class, high_range=max_class, memrate=memrate
            )
        else:
            x_train, y_train = self._select(
                self.data_train, self.targets_train, low_range=min_class, high_range=max_class
            )            
        nb_new_classes = len(np.unique(y_train))
        x_val, y_val = self._select(
            self.data_val, self.targets_val, low_range=min_class, high_range=max_class
        )
        if self._all_test_classes is True:
            logger.info("Balanced subset: Testing on all classes!")
            x_test, y_test = self._select(
                self.data_test, self.targets_test, high_range=sum(self.increments)
            )
        elif self._all_test_classes is not None or self._all_test_classes is not False:
            max_class = sum(self.increments[:self._current_task + 1 + self._all_test_classes])
            logger.info(
                f"Balanced subset: Testing on {self._all_test_classes} unseen tasks (max class = {max_class})."
            )
            x_test, y_test = self._select(self.data_test, self.targets_test, high_range=max_class)
        else:
            x_test, y_test = self._select(self.data_test, self.targets_test, high_range=max_class)

        if self._onehot:

            def to_onehot(x):
                n = np.max(x) + 1
                return np.eye(n)[x]

            y_train = to_onehot(y_train)

        logger.info("Balanced subset: Set memory of size {}.".format(memory[0].shape[0]))
        data_val_new, targets_val_new, val_memory_flags, \
            data_train_new, targets_train_new, train_memory_flags \
             = self._add_searching_memory(min_class, max_class, x_train, y_train, *memory)

        #import pdb
        #pdb.set_trace()
        train_loader = self._get_loader(data_train_new, targets_train_new, train_memory_flags, mode="train")
        val_loader = None
        test_loader = self._get_loader(data_val_new, targets_val_new, np.zeros((data_val_new.shape[0],)), mode="test")

        task_info = {
            "min_class": min_class,
            "max_class": max_class,
            "total_n_classes": sum(self.increments),
            "increment": nb_new_classes,  # self.increments[self._current_task],
            "task": self._current_task,
            "max_task": len(self.increments),
            "n_train_data": x_train.shape[0],
            "n_test_data": x_test.shape[0]
        }

        return task_info, train_loader, val_loader, test_loader

    def new_task(self, memory=None, memory_val=None, memrate=None, task_id=None, disable_rmm=None):
        if self._current_task >= len(self.increments):
            raise Exception("No more tasks.")

        min_class = sum(self.increments[:self._current_task])
        max_class = sum(self.increments[:self._current_task + 1])
        
        if task_id != 0 and not disable_rmm:
            x_train, y_train = self._select_train(
                self.data_train, self.targets_train, low_range=min_class, high_range=max_class, memrate=memrate
            )
        else:
            x_train, y_train = self._select(
                self.data_train, self.targets_train, low_range=min_class, high_range=max_class
            )            
        nb_new_classes = len(np.unique(y_train))
        x_val, y_val = self._select(
            self.data_val, self.targets_val, low_range=min_class, high_range=max_class
        )
        if self._all_test_classes is True:
            logger.info("Testing on all classes!")
            x_test, y_test = self._select(
                self.data_test, self.targets_test, high_range=sum(self.increments)
            )
        elif self._all_test_classes is not None or self._all_test_classes is not False:
            max_class = sum(self.increments[:self._current_task + 1 + self._all_test_classes])
            logger.info(
                f"Testing on {self._all_test_classes} unseen tasks (max class = {max_class})."
            )
            x_test, y_test = self._select(self.data_test, self.targets_test, high_range=max_class)
        else:
            x_test, y_test = self._select(self.data_test, self.targets_test, high_range=max_class)

        if self._onehot:

            def to_onehot(x):
                n = np.max(x) + 1
                return np.eye(n)[x]

            y_train = to_onehot(y_train)

        if memory is not None:
            logger.info("Set memory of size: {}.".format(memory[0].shape[0]))
            x_train, y_train, train_memory_flags = self._add_memory(x_train, y_train, *memory)
        else:
            train_memory_flags = np.zeros((x_train.shape[0],))
        if memory_val is not None:
            logger.info("Set validation memory of size: {}.".format(memory_val[0].shape[0]))
            x_val, y_val, val_memory_flags = self._add_memory(x_val, y_val, *memory_val)
        else:
            val_memory_flags = np.zeros((x_val.shape[0],))

        train_loader = self._get_loader(x_train, y_train, train_memory_flags, mode="train")
        val_loader = self._get_loader(x_val, y_val, val_memory_flags,
                                      mode="train") if len(x_val) > 0 else None
        test_loader = self._get_loader(x_test, y_test, np.zeros((x_test.shape[0],)), mode="test")

        task_info = {
            "min_class": min_class,
            "max_class": max_class,
            "total_n_classes": sum(self.increments),
            "increment": nb_new_classes,  # self.increments[self._current_task],
            "task": self._current_task,
            "max_task": len(self.increments),
            "n_train_data": x_train.shape[0],
            "n_test_data": x_test.shape[0]
        }

        self._current_task += 1

        return task_info, train_loader, val_loader, test_loader

    def _add_searching_memory(self, min_class, max_class, data_train, targets_train, data_memory, targets_memory):
        #if self._onehot: 
        #    targets_memory = np.concatenate(
        #        (
        #            targets_memory,
        #            np.zeros((targets_memory.shape[0], self.increments[self._current_task]))
        #        ),
        #        axis=1
        #    )

        selected_idxes = random.sample(range(len(targets_memory)),int(0.5*len(targets_memory)))
        data_val_memory = data_memory[selected_idxes]
        targets_val_memory = targets_memory[selected_idxes]

        mem_per_cls = len(targets_memory)//min_class
        num_of_val_new_cls = int(mem_per_cls*(max_class-min_class)/2)

        try:
            selected_new_idxes = random.sample(range(len(targets_train)),num_of_val_new_cls)
            data_val_new = data_train[selected_new_idxes]
        except:
            import pdb
            pdb.set_trace()
        targets_val_new = targets_train[selected_new_idxes]

        val_memory_flags = np.concatenate((np.zeros((data_val_new.shape[0],)), np.ones((data_val_memory.shape[0],))))

        remaining_indexes = np.array([i not in selected_idxes for i in range(len(data_memory))])
        data_train_memory = data_memory[remaining_indexes]
        targets_train_memory = targets_memory[remaining_indexes]

        remaining_new_indexes = np.array([i not in selected_new_idxes for i in range(len(data_train))])
        data_train_new = data_train[remaining_new_indexes]
        targets_train_new = targets_train[remaining_new_indexes]

        train_memory_flags = np.concatenate((np.zeros((data_train_new.shape[0],)), np.ones((data_train_memory.shape[0],))))

        data_val_new = np.concatenate((data_val_new, data_val_memory))
        targets_val_new = np.concatenate((targets_val_new, targets_val_memory))

        data_train_new = np.concatenate((data_train_new, data_train_memory))
        targets_train_new = np.concatenate((targets_train_new, targets_train_memory))

        return data_val_new, targets_val_new, val_memory_flags, data_train_new, targets_train_new, train_memory_flags

    def _add_memory(self, x, y, data_memory, targets_memory):
        if self._onehot: 
            targets_memory = np.concatenate(
                (
                    targets_memory,
                    np.zeros((targets_memory.shape[0], self.increments[self._current_task]))
                ),
                axis=1
            )

        memory_flags = np.concatenate((np.zeros((x.shape[0],)), np.ones((data_memory.shape[0],))))

        x = np.concatenate((x, data_memory))
        y = np.concatenate((y, targets_memory))

        return x, y, memory_flags

    def get_custom_loader_compressed(
        self, class_indexes, memory=None, mode="test", data_source="train", sampler=None
    ):
        if not isinstance(class_indexes, list):  # TODO: deprecated, should always give a list
            class_indexes = [class_indexes]

        if data_source == "train":
            x, y = self.data_train, self.targets_train
        elif data_source == "val":
            x, y = self.data_val, self.targets_val
        elif data_source == "test":
            x, y = self.data_test, self.targets_test
        else:
            raise ValueError("Unknown data source <{}>.".format(data_source))

        data, targets = [], []
        for class_index in class_indexes:
            class_data, class_targets = self._select(
                x, y, low_range=class_index, high_range=class_index + 1
            )
            data.append(class_data)
            targets.append(class_targets)

        if len(data) == 0:
            assert memory is not None
        else:
            data = np.concatenate(data)
            targets = np.concatenate(targets)

        if (not isinstance(memory, tuple) and
            memory is not None) or (isinstance(memory, tuple) and memory[0] is not None):
            if len(data) > 0:
                data, targets, memory_flags = self._add_memory(data, targets, *memory)
            else:
                data, targets = memory
                memory_flags = np.ones((data.shape[0],))
        else:
            memory_flags = np.zeros((data.shape[0],))

        return data, self._get_loader(
            data, targets, memory_flags, shuffle=False, mode=mode, sampler=sampler
        )

    def get_custom_loader_compressed_2(
        self, class_indexes, memory=None, mode="test", data_source="train", sampler=None
    ):
        if not isinstance(class_indexes, list):  # TODO: deprecated, should always give a list
            class_indexes = [class_indexes]

        if data_source == "train":
            x, y = self.data_train, self.targets_train
        elif data_source == "val":
            x, y = self.data_val, self.targets_val
        elif data_source == "test":
            x, y = self.data_test, self.targets_test
        else:
            raise ValueError("Unknown data source <{}>.".format(data_source))

        the_compressed_img_path = './compress_imgs/imagenet_sub'

        data, targets = [], []
        for class_index in class_indexes:
            class_data, class_targets = self._select(
                x, y, low_range=class_index, high_range=class_index + 1
            )
            cls_label = class_data[0][20:29]
            compressed_img_dir = the_compressed_img_path + '/' + cls_label
            compressed_imgs = glob.glob(compressed_img_dir+'/*')
            compressed_imgs = np.array(compressed_imgs)

            data.append(compressed_imgs)
            targets.append(class_targets[0:len(compressed_imgs)])

        if len(data) == 0:
            assert memory is not None
        else:
            data = np.concatenate(data)
            targets = np.concatenate(targets)

        if (not isinstance(memory, tuple) and
            memory is not None) or (isinstance(memory, tuple) and memory[0] is not None):
            if len(data) > 0:
                data, targets, memory_flags = self._add_memory(data, targets, *memory)
            else:
                data, targets = memory
                memory_flags = np.ones((data.shape[0],))
        else:
            memory_flags = np.zeros((data.shape[0],))

        return data, self._get_loader(
            data, targets, memory_flags, shuffle=False, mode=mode, sampler=sampler
        )

    def get_custom_loader(
        self, class_indexes, memory=None, mode="test", data_source="train", sampler=None
    ):
        if not isinstance(class_indexes, list): 
            class_indexes = [class_indexes]

        if data_source == "train":
            x, y = self.data_train, self.targets_train
        elif data_source == "val":
            x, y = self.data_val, self.targets_val
        elif data_source == "test":
            x, y = self.data_test, self.targets_test
        else:
            raise ValueError("Unknown data source <{}>.".format(data_source))

        data, targets = [], []
        for class_index in class_indexes:
            class_data, class_targets = self._select(
                x, y, low_range=class_index, high_range=class_index + 1
            )
            data.append(class_data)
            targets.append(class_targets)

        if len(data) == 0:
            assert memory is not None
        else:
            data = np.concatenate(data)
            targets = np.concatenate(targets)

        if (not isinstance(memory, tuple) and
            memory is not None) or (isinstance(memory, tuple) and memory[0] is not None):
            if len(data) > 0:
                data, targets, memory_flags = self._add_memory(data, targets, *memory)
            else:
                data, targets = memory
                memory_flags = np.ones((data.shape[0],))
        else:
            memory_flags = np.zeros((data.shape[0],))

        return data, self._get_loader(
            data, targets, memory_flags, shuffle=False, mode=mode, sampler=sampler
        )

    def get_memory_loader(self, data, targets):
        return self._get_loader(
            data, targets, np.ones((data.shape[0],)), shuffle=True, mode="train"
        )

    def _select_train(self, x, y, low_range=0, high_range=0, memrate=None):
        if memrate != 0:
            idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
            #selected_idxes = np.random.randint(0, len(idxes),size=int((1-memrate)*len(idxes)))
            selected_idxes = random.sample(range(len(idxes)),int((1-memrate)*len(idxes)))
            new_idxes = idxes[selected_idxes]
            new_idxes = np.sort(new_idxes)
        else:
            new_idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[new_idxes], y[new_idxes]

    def _select(self, x, y, low_range=0, high_range=0):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes]

    def _get_loader(self, x, y, memory_flags, shuffle=True, mode="train", sampler=None):
        if mode == "train":
            trsf = transforms.Compose([*self.train_transforms, *self.common_transforms])
        elif mode == "test":
            trsf = transforms.Compose([*self.test_transforms, *self.common_transforms])
        elif mode == "flip":
            trsf = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=1.), *self.test_transforms,
                    *self.common_transforms
                ]
            )
        else:
            raise NotImplementedError("Unknown mode {}.".format(mode))

        sampler = sampler or self._sampler
        if sampler is not None and mode == "train":
            logger.info("Using sampler {}".format(sampler))
            sampler = sampler(y, memory_flags, batch_size=self._batch_size, **self._sampler_config)
            batch_size = 1
        else:
            sampler = None
            batch_size = self._batch_size

        return DataLoader(
            DummyDataset(x, y, memory_flags, trsf, open_image=self.open_image),
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            num_workers=self._workers,
            batch_sampler=sampler
        )

    def _setup_data(
        self,
        datasets,
        random_order=False,
        class_order=None,
        seed=1,
        increment=10,
        validation_split=0.,
        initial_increment=None,
        data_path="data"
    ):
        self.data_train, self.targets_train = [], []
        self.data_test, self.targets_test = [], []
        self.data_val, self.targets_val = [], []
        self.increments = []
        self.class_order = []

        current_class_idx = 0 
        for dataset in datasets:
            train_dataset = dataset().base_dataset(data_path, train=True, download=True)
            test_dataset = dataset().base_dataset(data_path, train=False, download=True)

            x_train, y_train = train_dataset.data, np.array(train_dataset.targets)
            x_val, y_val, x_train, y_train = self._split_per_class(
                x_train, y_train, validation_split
            )
            x_test, y_test = test_dataset.data, np.array(test_dataset.targets)

            order = list(range(len(np.unique(y_train))))
            if random_order:
                random.seed(seed) 
                random.shuffle(order)
            elif class_order:
                order = class_order
            elif dataset.class_order is not None:
                order = dataset.class_order
            elif train_dataset.class_order is not None:
                order = train_dataset.class_order

            logger.info("Dataset {}: class ordering: {}.".format(dataset.__name__, order))

            self.class_order.append(order)

            y_train = self._map_new_class_index(y_train, order)
            y_val = self._map_new_class_index(y_val, order)
            y_test = self._map_new_class_index(y_test, order)


            y_train += current_class_idx
            y_val += current_class_idx
            y_test += current_class_idx

            current_class_idx += len(order)
            if len(datasets) > 1:
                self.increments.append(len(order))
            elif initial_increment is None:
                nb_steps = len(order) / increment
                remainder = len(order) - int(nb_steps) * increment

                if not nb_steps.is_integer():
                    logger.warning(
                        f"THe last step will have sligthly less sample ({remainder} vs {increment})."
                    )
                    self.increments = [increment for _ in range(int(nb_steps))]
                    self.increments.append(remainder)
                else:
                    self.increments = [increment for _ in range(int(nb_steps))]
            else:
                self.increments = [initial_increment]

                nb_steps = (len(order) - initial_increment) / increment
                remainder = (len(order) - initial_increment) - int(nb_steps) * increment
                if not nb_steps.is_integer():
                    logger.warning(
                        f"THe last step will have sligthly less sample ({remainder} vs {increment})."
                    )
                    self.increments.extend([increment for _ in range(int(nb_steps))])
                    self.increments.append(remainder)
                else:
                    self.increments.extend([increment for _ in range(int(nb_steps))])

            self.data_train.append(x_train)
            self.targets_train.append(y_train)
            self.data_val.append(x_val)
            self.targets_val.append(y_val)
            self.data_test.append(x_test)
            self.targets_test.append(y_test)

        self.data_train = np.concatenate(self.data_train)
        self.targets_train = np.concatenate(self.targets_train)
        self.data_val = np.concatenate(self.data_val)
        self.targets_val = np.concatenate(self.targets_val)
        self.data_test = np.concatenate(self.data_test)
        self.targets_test = np.concatenate(self.targets_test)

    @staticmethod
    def _map_new_class_index(y, order):
        return np.array(list(map(lambda x: order.index(x), y)))

    @staticmethod
    def _split_per_class(x, y, validation_split=0.):
        shuffled_indexes = np.random.permutation(x.shape[0])
        x = x[shuffled_indexes]
        y = y[shuffled_indexes]

        x_val, y_val = [], []
        x_train, y_train = [], []

        for class_id in np.unique(y):
            class_indexes = np.where(y == class_id)[0]
            nb_val_elts = int(class_indexes.shape[0] * validation_split)

            val_indexes = class_indexes[:nb_val_elts]
            train_indexes = class_indexes[nb_val_elts:]

            x_val.append(x[val_indexes])
            y_val.append(y[val_indexes])
            x_train.append(x[train_indexes])
            y_train.append(y[train_indexes])

        x_train, y_train = np.concatenate(x_train), np.concatenate(y_train)
        x_val, y_val = np.concatenate(x_val), np.concatenate(y_val)
        return x_val, y_val, x_train, y_train


class DummyDataset(torch.utils.data.Dataset):

    def __init__(self, x, y, memory_flags, trsf, open_image=False):
        self.x, self.y = x, y
        self.memory_flags = memory_flags
        self.trsf = trsf
        self.open_image = open_image

        assert x.shape[0] == y.shape[0] == memory_flags.shape[0]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]
        memory_flag = self.memory_flags[idx]

        if self.open_image:
            img = Image.open(x).convert("RGB")
        else:
            img = Image.fromarray(x.astype("uint8"))

        img = self.trsf(img)
        return {"inputs": img, "targets": y, "memory_flags": memory_flag}


def _get_datasets(dataset_names):
    return [_get_dataset(dataset_name) for dataset_name in dataset_names.split("-")]


def _get_dataset(dataset_name):
    dataset_name = dataset_name.lower().strip()

    if dataset_name == "cifar10":
        return iCIFAR10
    elif dataset_name == "cifar100":
        return iCIFAR100
    elif dataset_name == "imagenet100":
        return ImageNet100
    elif dataset_name == "imagenet100ucir":
        return ImageNet100UCIR
    elif dataset_name == "imagenet1000":
        return ImageNet1000
    elif dataset_name == "tinyimagenet":
        return TinyImageNet200
    elif dataset_name == "awa2":
        return AwA2
    elif dataset_name == "cub200":
        return CUB200
    elif dataset_name == "apy":
        return APY
    elif dataset_name == "lad":
        return LAD
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))
