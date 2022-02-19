import collections
import copy
import logging
import os
import pickle

import numpy as np
import torch
from scipy.spatial.distance import cdist
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from scipy.stats import entropy
from scipy.special import softmax

from inclearn.lib import factory, herding, losses, network, schedulers, utils
from inclearn.lib.network import hook
from inclearn.models.base import IncrementalLearner

EPSILON = 1e-8

logger = logging.getLogger(__name__)


class NewIncrementalLearner(IncrementalLearner):

    def __init__(self, args):
        super().__init__()

        self._disable_progressbar = args.get("no_progressbar", False)

        self._device = args["device"][0]
        self._debug = args["debug_mode"]
        self._using_compressed_exemplar = args["using_compressed_exemplar"]
        self._multiple_devices = args["device"]

        self._opt_name = args["optimizer"]
        self._lr = args["lr"]
        self._weight_decay = args["weight_decay"]
        self._n_epochs = args["epochs"]

        self._scheduling = args["scheduling"]
        self._lr_decay = args["lr_decay"]

        self._warmup_config = args.get("warmup", {})
        if self._warmup_config and self._warmup_config["total_epoch"] > 0:
            self._lr /= self._warmup_config["multiplier"]

        self._eval_every_x_epochs = args.get("eval_every_x_epochs")
        self._early_stopping = args.get("early_stopping", {})

        self._memory_size = args["memory_size"]
        self._fixed_memory = args["fixed_memory"]
        self._herding_selection = args.get("herding_selection", {"type": "icarl"})
        self._n_classes = 0
        self._last_results = None
        self._validation_percent = args["validation"]

        self._rotations_config = args.get("rotations_config", {})
        self._random_noise_config = args.get("random_noise_config", {})

        self._network = network.BasicNet(
            args["convnet"],
            convnet_kwargs=args.get("convnet_config", {}),
            classifier_kwargs=args.get("classifier_config", {
                "type": "fc",
                "use_bias": True
            }),
            device=self._device,
            extract_no_act=True,
            classifier_no_act=False,
            rotations_predictor=bool(self._rotations_config)
        )

        self._examplars = {}
        self._means = None
        self._herding_indexes = []
        self._data_memory, self._targets_memory = None, None

        self._old_model = None

        self._clf_loss = F.binary_cross_entropy_with_logits
        self._distil_loss = F.binary_cross_entropy_with_logits

        self._epoch_metrics = collections.defaultdict(list)

        self._meta_transfer = args.get("meta_transfer", {})

    def set_meta_transfer(self):
        if self._meta_transfer["type"] not in ("repeat", "once", "none"):
            raise ValueError(f"Invalid value for meta-transfer {self._meta_transfer}.")

        if self._task == 0:
            self._network.convnet.apply_mtl(False)
        if self._task > 0:
            self._network.convnet.apply_mtl(True)
            self._network.convnet.freeze_convnet(True)

    def save_metadata(self, directory, run_id):
        path = os.path.join(directory, f"meta_{run_id}_task_{self._task}.pkl")

        logger.info("Saving metadata at {}.".format(path))
        with open(path, "wb+") as f:
            pickle.dump(
                [self._data_memory, self._targets_memory, self._herding_indexes, self._class_means],
                f
            )

    def load_metadata(self, directory, run_id):
        path = os.path.join(directory, f"meta_{run_id}_task_{self._task}.pkl")
        if not os.path.exists(path):
            return

        logger.info("Loading metadata at {}.".format(path))
        with open(path, "rb") as f:
            self._data_memory, self._targets_memory, self._herding_indexes, self._class_means = pickle.load(
                f
            )

    @property
    def epoch_metrics(self):
        return dict(self._epoch_metrics)

    def _before_task(self, train_loader, val_loader):
        self._n_classes += self._task_size
        self._network.add_classes(self._task_size)
        logger.info("Now {} examplars per class.".format(self._memory_per_class))

        self._optimizer = factory.get_optimizer(
            self._network.parameters(), self._opt_name, self._lr, self._weight_decay
        )

        base_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self._optimizer, self._scheduling, gamma=self._lr_decay
        )

        if self._warmup_config:
            if self._warmup_config.get("only_first_step", True) and self._task != 0:
                pass
            else:
                logger.info("Using WarmUp")
                self._scheduler = schedulers.GradualWarmupScheduler(
                    optimizer=self._optimizer,
                    after_scheduler=base_scheduler,
                    **self._warmup_config
                )
        else:
            self._scheduler = base_scheduler

    def _train_task(self, train_loader, val_loader):
        logger.debug("nb {}.".format(len(train_loader.dataset)))
        self._training_step(train_loader, val_loader, 0, self._n_epochs)

    def _training_step(
        self, train_loader, val_loader, initial_epoch, nb_epochs, record_bn=True, clipper=None
    ):
        best_epoch, best_acc = -1, -1.
        wait = 0

        grad, act = None, None
        if len(self._multiple_devices) > 1:
            logger.info("Duplicating model on {} gpus.".format(len(self._multiple_devices)))
            training_network = nn.DataParallel(self._network, self._multiple_devices)
            if self._network.gradcam_hook:
                grad, act, back_hook, for_hook = hook.get_gradcam_hook(training_network)
                training_network.module.convnet.last_conv.register_backward_hook(back_hook)
                training_network.module.convnet.last_conv.register_forward_hook(for_hook)
        else:
            training_network = self._network

        for epoch in range(initial_epoch, nb_epochs):
            self._metrics = collections.defaultdict(float)

            self._epoch_percent = epoch / (nb_epochs - initial_epoch)

            prog_bar = tqdm(
                train_loader,
                disable=self._disable_progressbar,
                ascii=True,
                bar_format="{desc}: {percentage:3.0f}% | {n_fmt}/{total_fmt} | {rate_fmt}{postfix}"
            )
            for i, input_dict in enumerate(prog_bar, start=1):
                if self._debug:
                    if i > 1:
                        break
                inputs, targets = input_dict["inputs"], input_dict["targets"]
                memory_flags = input_dict["memory_flags"]

                if grad is not None:
                    _clean_list(grad)
                    _clean_list(act)

                self._optimizer.zero_grad()
                loss = self._forward_loss(
                    training_network,
                    inputs,
                    targets,
                    memory_flags,
                    gradcam_grad=grad,
                    gradcam_act=act
                )
                loss.backward()
                self._optimizer.step()

                if clipper:
                    training_network.apply(clipper)

                self._print_metrics(prog_bar, epoch, nb_epochs, i)

            if self._scheduler:
                self._scheduler.step(epoch)

            if self._eval_every_x_epochs and epoch != 0 and epoch % self._eval_every_x_epochs == 0:
                self._network.eval()
                self._data_memory, self._targets_memory, self._herding_indexes, self._class_means = self.build_examplars(
                    self.inc_dataset, self._herding_indexes
                )
                ytrue, ypred = self._eval_task(val_loader)
                acc = 100 * round((ypred == ytrue).sum() / len(ytrue), 3)
                logger.info("Val accuracy: {}".format(acc))
                self._network.train()

                if acc > best_acc:
                    best_epoch = epoch
                    best_acc = acc
                    wait = 0
                else:
                    wait += 1

                if self._early_stopping and self._early_stopping["patience"] > wait:
                    logger.warning("Early stopping!")
                    break

        if self._eval_every_x_epochs:
            logger.info("Best accuracy reached at epoch {} with {}%.".format(best_epoch, best_acc))

    def _print_metrics(self, prog_bar, epoch, nb_epochs, nb_batches):
        pretty_metrics = ", ".join(
            "{}: {}".format(metric_name, round(metric_value / nb_batches, 3))
            for metric_name, metric_value in self._metrics.items()
        )

        prog_bar.set_description(
            "T{}/{}, E{}/{} => {}".format(
                self._task + 1, self._n_tasks, epoch + 1, nb_epochs, pretty_metrics
            )
        )

    def _forward_loss(
        self,
        training_network,
        inputs,
        targets,
        memory_flags,
        gradcam_grad=None,
        gradcam_act=None,
        **kwargs
    ):
        inputs, targets = inputs.to(self._device), targets.to(self._device)
        onehot_targets = utils.to_onehot(targets, self._n_classes).to(self._device)

        outputs = training_network(inputs)
        if gradcam_act is not None:
            outputs["gradcam_gradients"] = gradcam_grad
            outputs["gradcam_activations"] = gradcam_act

        loss = self._compute_loss(inputs, outputs, targets, onehot_targets, memory_flags)

        if not utils.check_loss(loss):
            raise ValueError("A loss is NaN: {}".format(self._metrics))

        self._metrics["loss"] += loss.item()

        return loss

    def _after_task_intensive(self, inc_dataset):
        if self._herding_selection["type"] == "confusion":
            self._compute_confusion_matrix()

        self._data_memory, self._targets_memory, self._herding_indexes, self._class_means = self.build_examplars(
            inc_dataset, self._herding_indexes
        )

    def _after_task(self, inc_dataset):
        self._old_model = self._network.copy().freeze().to(self._device)
        self._network.on_task_end()

    def _compute_confusion_matrix(self):
        use_validation = self._validation_percent > 0.
        _, loader = self.inc_dataset.get_custom_loader(
            list(range(self._n_classes - self._task_size, self._n_classes)),
            memory=self.get_val_memory() if use_validation else self.get_memory(),
            mode="test",
            data_source="val" if use_validation else "train"
        )
        ypreds, ytrue = self._eval_task(loader)
        self._last_results = (ypreds, ytrue)

    def plot_tsne(self):
        if self.folder_result:
            loader = self.inc_dataset.get_custom_loader([], memory=self.get_memory())[1]
            embeddings, targets = utils.extract_features(self._network, loader)
            utils.plot_tsne(
                os.path.join(self.folder_result, "tsne_{}".format(self._task)), embeddings, targets
            )

    def _eval_task(self, data_loader):
        ypreds, ytrue = self.compute_accuracy(self._network, data_loader, self._class_means)

        return ypreds, ytrue

    def _compute_loss(self, inputs, outputs, targets, onehot_targets, memory_flags):
        logits = outputs["logits"]

        if self._old_model is None:
            loss = F.binary_cross_entropy_with_logits(logits, onehot_targets)
        else:
            with torch.no_grad():
                old_targets = torch.sigmoid(self._old_model(inputs)["logits"])

            new_targets = onehot_targets.clone()
            new_targets[..., :-self._task_size] = old_targets

            loss = F.binary_cross_entropy_with_logits(logits, new_targets)

        if self._rotations_config:
            rotations_loss = losses.unsupervised_rotations(
                inputs, memory_flags, self._network, self._rotations_config
            )
            loss += rotations_loss
            self._metrics["rot"] += rotations_loss.item()

        return loss

    @property
    def _memory_per_class(self):
        """Returns the number of examplars per class."""
        if self._fixed_memory:
            return self._memory_size // self._total_n_classes
        return self._memory_size // self._n_classes


    def build_examplars(
        self, inc_dataset, herding_indexes, memory_per_class=None, data_source="train"
    ):
        logger.info("Building & updating memory.")
        memory_per_class = memory_per_class or self._memory_per_class
        herding_indexes = copy.deepcopy(herding_indexes)

        data_memory, targets_memory = [], []
        class_means = np.zeros((self._n_classes, self._network.features_dim))

        all_cls_entropies = []
        for class_idx in range(self._n_classes):
            inputs, loader = inc_dataset.get_custom_loader(class_idx, mode="test", data_source=data_source)

            features, targets, predictions = utils.extract_features_and_predictions(self._network, loader)

            the_entropies = [entropy(softmax(this_item)) for this_item in predictions]
            mean_entropy = np.mean(the_entropies)
            all_cls_entropies.append(mean_entropy)
        entropy_median = np.median(all_cls_entropies)
        for the_entropy in all_cls_entropies:
            if the_entropy > entropy_median:
                self._cls_group_idx.append(1)
            else:
                self._cls_group_idx.append(0)

        memory_per_class_list = [int(memory_per_class*(1-self._current_cls_rate)), int(memory_per_class*(1+self._current_cls_rate))]

        for class_idx in range(self._n_classes):
            this_cls_memory_per_class = memory_per_class_list[self._cls_group_idx[class_idx]]
            inputs, loader = inc_dataset.get_custom_loader(class_idx, mode="test", data_source=data_source)               

            features, targets = utils.extract_features(self._network, loader)
            features_flipped, _ = utils.extract_features(
                self._network,
                inc_dataset.get_custom_loader(class_idx, mode="flip", data_source=data_source)[1]
            )

            if class_idx >= self._n_classes - self._task_size:
                if self._herding_selection["type"] == "icarl":
                    selected_indexes = herding.icarl_selection(features, this_cls_memory_per_class)
                elif self._herding_selection["type"] == "closest":
                    selected_indexes = herding.closest_to_mean(features, this_cls_memory_per_class)
                elif self._herding_selection["type"] == "random":
                    selected_indexes = herding.random(features, this_cls_memory_per_class)
                elif self._herding_selection["type"] == "first":
                    selected_indexes = np.arange(this_cls_memory_per_class)
                elif self._herding_selection["type"] == "kmeans":
                    selected_indexes = herding.kmeans(
                        features, this_cls_memory_per_class, k=self._herding_selection["k"]
                    )
                elif self._herding_selection["type"] == "confusion":
                    selected_indexes = herding.confusion(
                        *self._last_results,
                        this_cls_memory_per_class,
                        class_id=class_idx,
                        minimize_confusion=self._herding_selection["minimize_confusion"]
                    )
                elif self._herding_selection["type"] == "var_ratio":
                    selected_indexes = herding.var_ratio(
                        this_cls_memory_per_class, self._network, loader, **self._herding_selection
                    )
                elif self._herding_selection["type"] == "mcbn":
                    selected_indexes = herding.mcbn(
                        this_cls_memory_per_class, self._network, loader, **self._herding_selection
                    )
                else:
                    raise ValueError(
                        "Unknown herding selection {}.".format(self._herding_selection)
                    )

                herding_indexes.append(selected_indexes)

            if class_idx < self._n_classes - self._task_size:
                selected_indexes = herding_indexes[class_idx][:this_cls_memory_per_class]
                herding_indexes[class_idx] = selected_indexes

            examplar_mean = self.compute_examplar_mean(
                features, features_flipped, selected_indexes, this_cls_memory_per_class
            )

            data_memory.append(inputs[selected_indexes])
            targets_memory.append(targets[selected_indexes])

            class_means[class_idx, :] = examplar_mean

        data_memory = np.concatenate(data_memory)
        targets_memory = np.concatenate(targets_memory)

        return data_memory, targets_memory, herding_indexes, class_means

    def get_memory(self):
        return self._data_memory, self._targets_memory

    @staticmethod
    def compute_examplar_mean(feat_norm, feat_flip, indexes, nb_max):
        D = feat_norm.T
        D = D / (np.linalg.norm(D, axis=0) + EPSILON)

        D2 = feat_flip.T
        D2 = D2 / (np.linalg.norm(D2, axis=0) + EPSILON)

        selected_d = D[..., indexes]
        selected_d2 = D2[..., indexes]

        mean = (np.mean(selected_d, axis=1) + np.mean(selected_d2, axis=1)) / 2
        mean /= (np.linalg.norm(mean) + EPSILON)

        return mean

    @staticmethod
    def compute_accuracy(model, loader, class_means):
        features, targets_ = utils.extract_features(model, loader)

        features = (features.T / (np.linalg.norm(features.T, axis=0) + EPSILON)).T

        sqd = cdist(class_means, features, 'sqeuclidean')
        score_icarl = (-sqd).T

        return score_icarl, targets_


def _clean_list(l):
    for i in range(len(l)):
        l[i] = None
