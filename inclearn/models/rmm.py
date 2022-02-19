import copy
import logging
import math

import numpy as np
import torch
from torch.nn import functional as F

from inclearn.lib import data, factory, losses, network, utils
from inclearn.lib.data import samplers
from inclearn.models.new_base import NewIncrementalLearner

logger = logging.getLogger(__name__)


class RMMLearner(NewIncrementalLearner):

    def __init__(self, args):
        self._disable_progressbar = args.get("no_progressbar", False)

        self._device = args["device"][0]
        self._multiple_devices = args["device"]
        self._debug = args["debug_mode"]
        self._using_compressed_exemplar = args["using_compressed_exemplar"]
        
        self._batch_size = args["batch_size"]
        self._opt_name = args["optimizer"]
        self._lr = args["lr"]
        self._weight_decay = args["weight_decay"]
        self._n_epochs = args["epochs"]
        self._scheduling = args["scheduling"]
        self._lr_decay = args["lr_decay"]

        self._memory_size = args["memory_size"]
        self._fixed_memory = args["fixed_memory"]
        self._disable_rmm = args["disable_rmm"]
        self._herding_selection = args.get("herding_selection", {"type": "icarl"})
        self._n_classes = 0
        self._last_results = None
        self._validation_percent = args.get("validation")
        self._current_mem_rate = None

        self._weight_pod_loss = args["weight_pod_loss"]
        self._weight_icarl_loss = args["weight_icarl_loss"]
        self._weight_lucir_loss = args["weight_lucir_loss"]

        self._pod_flat_config = args.get("pod_flat", {})
        self._pod_spatial_config = args.get("pod_spatial", {})

        self._nca_config = args.get("nca", {})
        self._softmax_ce = args.get("softmax_ce", False)

        self._perceptual_features = args.get("perceptual_features")
        self._perceptual_style = args.get("perceptual_style")

        self._groupwise_factors = args.get("groupwise_factors", {})
        self._groupwise_factors_bis = args.get("groupwise_factors_bis", {})

        self._class_weights_config = args.get("class_weights_config", {})

        self._evaluation_type = args.get("eval_type", "icarl")
        self._evaluation_config = args.get("evaluation_config", {})

        self._eval_every_x_epochs = args.get("eval_every_x_epochs")
        self._early_stopping = args.get("early_stopping", {})

        self._gradcam_distil = args.get("gradcam_distil", {})

        classifier_kwargs = args.get("classifier_config", {})
        self.classifier_kwargs = classifier_kwargs
        self._network = network.BasicNet(
            args["convnet"],
            convnet_kwargs=args.get("convnet_config", {}),
            classifier_kwargs=classifier_kwargs,
            postprocessor_kwargs=args.get("postprocessor_config", {}),
            device=self._device,
            return_features=True,
            extract_no_act=True,
            classifier_no_act=args.get("classifier_no_act", True),
            attention_hook=True,
            gradcam_hook=bool(self._gradcam_distil)
        )

        self._examplars = {}
        self._means = None

        self._old_model = None

        self._finetuning_config = args.get("finetuning_config")

        self._herding_indexes = []
        self._cls_group_idx = []

        self._weight_generation = args.get("weight_generation")

        self._post_processing_type = None
        self._data_memory, self._targets_memory = None, None

        self._args = args
        self._args["_logs"] = {}

    @property
    def _memory_per_class(self):
        """Returns the number of examplars per class."""

        if self._args["dataset"]=='cifar100':
            img_pre_cls = 500
        else:
            img_pre_cls = 1300

        if self._fixed_memory:
            base_exemplar_budget = self._args["memory_size"]
        else:
            base_exemplar_budget = self._args["memory_size"]//self._total_n_classes * self._n_classes

        if self._disable_rmm:
            self._memory_size = base_exemplar_budget
        else:
            self._memory_size = base_exemplar_budget + int(self._current_mem_rate*self._args["increment"]*img_pre_cls)

        return self._memory_size // self._n_classes

    def _train_task(self, train_loader, val_loader):

        logger.info("PODNetLoss: " + str(self._weight_pod_loss))
        logger.info("iCaRLLoss: " + str(self._weight_icarl_loss))
        logger.info("LUCIRLoss: " + str(self._weight_lucir_loss))

        for p in self._network.parameters():
            if p.requires_grad:
                p.register_hook(lambda grad: torch.clamp(grad, -5., 5.))

        logger.debug("nb {}.".format(len(train_loader.dataset)))

        clipper = None
        self._training_step(
            train_loader, val_loader, 0, self._n_epochs, record_bn=True, clipper=clipper
        )

        self._post_processing_type = None

        if self._finetuning_config and self._task != 0:
            logger.info("Fine-tuning")
            if self._finetuning_config["scaling"]:
                logger.info(
                    "Custom fine-tuning scaling of {}.".format(self._finetuning_config["scaling"])
                )
                self._post_processing_type = self._finetuning_config["scaling"]

            if self._finetuning_config["sampling"] == "undersampling":
                self._data_memory, self._targets_memory, _, _ = self.build_examplars(
                    self.inc_dataset, self._herding_indexes
                )
                loader = self.inc_dataset.get_memory_loader(*self.get_memory())
            elif self._finetuning_config["sampling"] == "oversampling":
                _, loader = self.inc_dataset.get_custom_loader(
                    list(range(self._n_classes - self._task_size, self._n_classes)),
                    memory=self.get_memory(),
                    mode="train",
                    sampler=samplers.MemoryOverSampler
                )

            if self._finetuning_config["tuning"] == "all":
                parameters = self._network.parameters()
            elif self._finetuning_config["tuning"] == "convnet":
                parameters = self._network.convnet.parameters()
            elif self._finetuning_config["tuning"] == "classifier":
                parameters = self._network.classifier.parameters()
            elif self._finetuning_config["tuning"] == "classifier_scale":
                parameters = [
                    {
                        "params": self._network.classifier.parameters(),
                        "lr": self._finetuning_config["lr"]
                    }, {
                        "params": self._network.post_processor.parameters(),
                        "lr": self._finetuning_config["lr"]
                    }
                ]
            else:
                raise NotImplementedError(
                    "Unknwown finetuning parameters {}.".format(self._finetuning_config["tuning"])
                )

            self._optimizer = factory.get_optimizer(
                parameters, self._opt_name, self._finetuning_config["lr"], self.weight_decay
            )
            self._scheduler = None
            self._training_step(
                loader,
                val_loader,
                self._n_epochs,
                self._n_epochs + self._finetuning_config["epochs"],
                record_bn=False
            )

    @property
    def weight_decay(self):
        if isinstance(self._weight_decay, float):
            return self._weight_decay
        elif isinstance(self._weight_decay, dict):
            start, end = self._weight_decay["start"], self._weight_decay["end"]
            step = (max(start, end) - min(start, end)) / (self._n_tasks - 1)
            factor = -1 if start > end else 1

            return start + factor * self._task * step
        raise TypeError(
            "Invalid type {} for weight decay: {}.".format(
                type(self._weight_decay), self._weight_decay
            )
        )

    def _after_task(self, inc_dataset):
        if self._gradcam_distil:
            self._network.zero_grad()
            self._network.unset_gradcam_hook()
            self._old_model = self._network.copy().eval().to(self._device)
            self._network.on_task_end()

            self._network.set_gradcam_hook()
            self._old_model.set_gradcam_hook()
        else:
            super()._after_task(inc_dataset)

    def _eval_task(self, test_loader):
        if self._evaluation_type in ("icarl", "nme"):
            return super()._eval_task(test_loader)
        elif self._evaluation_type in ("softmax", "cnn"):
            ypred = []
            ytrue = []

            for input_dict in test_loader:
                ytrue.append(input_dict["targets"].numpy())

                inputs = input_dict["inputs"].to(self._device)
                logits = self._network(inputs)["logits"].detach()

                preds = F.softmax(logits, dim=-1)
                ypred.append(preds.cpu().numpy())

            ypred = np.concatenate(ypred)
            ytrue = np.concatenate(ytrue)

            self._last_results = (ypred, ytrue)

            return ypred, ytrue
        else:
            raise ValueError(self._evaluation_type)

    def _gen_weights(self):
        if self._weight_generation:
            utils.add_new_weights(
                self._network, self._weight_generation if self._task != 0 else "basic",
                self._n_classes, self._task_size, self.inc_dataset
            )

    def _before_task(self, train_loader, val_loader):
        if self._task == 1:
            print("Using weight transferring operation")
            self._network_1 = self._network.copy().to(self._device)
            self._network_2 = self._network.copy().to(self._device)
            the_convnet_kwargs = self._args.get("convnet_config", {})
            the_convnet_type = self._args["convnet"] + '_mtl'
            self._network_1.convnet  = factory.get_convnet(the_convnet_type, **the_convnet_kwargs)
            _temp_dict = self._network_2.convnet.state_dict()
            _network_dict = self._network_1.convnet.state_dict()
            _network_dict.update(_temp_dict)
            self._network_1.convnet.load_state_dict(_network_dict)
            self._network_1.convnet.to(self._device)
            self._network = network.AANets(
                self._network_1.convnet,
                self._network_2.convnet,
                classifier_kwargs=self.classifier_kwargs,
                postprocessor_kwargs=self._args.get("postprocessor_config", {}),
                device=self._device,
                return_features=True,
                extract_no_act=True,
                classifier_no_act=self._args.get("classifier_no_act", True),
                attention_hook=True,
                gradcam_hook=bool(self._gradcam_distil),
                dataset=self._args["dataset"]
            )
            self._network.classifier = self._network_1.classifier

        self._gen_weights()
        self._n_classes += self._task_size
        logger.info("Now {} examplars per class.".format(self._memory_per_class))

        if self._groupwise_factors and isinstance(self._groupwise_factors, dict):
            if self._groupwise_factors_bis and self._task > 0:
                logger.info("Using second set of groupwise lr.")
                groupwise_factor = self._groupwise_factors_bis
            else:
                groupwise_factor = self._groupwise_factors

            params = []
            for group_name, group_params in self._network.get_group_parameters().items():
                if group_params is None or group_name == "last_block" or group_name == "last_block_1" or group_name == "last_block_2":
                    continue
                factor = groupwise_factor.get(group_name, 1.0)
                if factor == 0.:
                    continue
                params.append({"params": group_params, "lr": self._lr * factor})
                print(f"Group: {group_name}, lr: {self._lr * factor}.")
        elif self._groupwise_factors == "ucir":
            params = [
                {
                    "params": self._network.convnet.parameters(),
                    "lr": self._lr
                },
                {
                    "params": self._network.classifier.new_weights,
                    "lr": self._lr
                },
            ]
        else:
            params = self._network.parameters()

        self._optimizer = factory.get_optimizer(params, self._opt_name, self._lr, self.weight_decay)

        self._scheduler = factory.get_lr_scheduler(
            self._scheduling,
            self._optimizer,
            nb_epochs=self._n_epochs,
            lr_decay=self._lr_decay,
            task=self._task
        )

        if self._class_weights_config:
            self._class_weights = torch.tensor(
                data.get_class_weights(train_loader.dataset, **self._class_weights_config)
            ).to(self._device)
        else:
            self._class_weights = None

    def _compute_loss(self, inputs, outputs, targets, onehot_targets, memory_flags):
        features, logits, atts = outputs["raw_features"], outputs["logits"], outputs["attention"]

        if self._post_processing_type is None:
            scaled_logits = self._network.post_process(logits)
        else:
            scaled_logits = logits * self._post_processing_type

        if self._old_model is not None:
            with torch.no_grad():
                old_outputs = self._old_model(inputs)
                old_features = old_outputs["raw_features"]
                old_atts = old_outputs["attention"]

        if self._nca_config:
            nca_config = copy.deepcopy(self._nca_config)
            if self._network.post_processor:
                nca_config["scale"] = self._network.post_processor.factor

            loss = losses.nca(
                logits,
                targets,
                memory_flags=memory_flags,
                class_weights=self._class_weights,
                **nca_config
            )
            self._metrics["nca"] += loss.item()
        elif self._softmax_ce:
            loss = F.cross_entropy(scaled_logits, targets)
            self._metrics["cce"] += loss.item()

        if self._old_model is not None:

            #import pdb
            #pdb.set_trace()

            # iCaRL loss
            with torch.no_grad():
                old_targets = torch.sigmoid(self._old_model(inputs)["logits"])

            new_targets = onehot_targets.clone()
            new_targets[..., :-self._task_size] = old_targets

            icarl_loss = F.binary_cross_entropy_with_logits(logits, new_targets)
            loss += self._weight_icarl_loss * icarl_loss

            # LUCIR loss
            self._less_forget_lambda = 5.0
            scheduled_lambda = self._less_forget_lambda * math.sqrt(self._n_classes / self._task_size)
            lessforget_loss = scheduled_lambda * losses.embeddings_similarity(old_features, features)
            loss += self._weight_lucir_loss * lessforget_loss

            self._ranking_loss_factor = 1.0
            self._ranking_loss_nb_negatives = 2
            self._ranking_loss_margin = 0.5

            ranking_loss = self._ranking_loss_factor * losses.ucir_ranking(logits, targets, self._n_classes, self._task_size, nb_negatives=min(self._ranking_loss_nb_negatives, self._task_size), margin=self._ranking_loss_margin)
            loss += self._weight_lucir_loss * ranking_loss

            # PODNet loss
            if self._pod_flat_config:
                if self._pod_flat_config["scheduled_factor"]:
                    factor = self._pod_flat_config["scheduled_factor"] * math.sqrt(
                        self._n_classes / self._task_size
                    )
                else:
                    factor = self._pod_flat_config.get("factor", 1.)

                pod_flat_loss = factor * losses.embeddings_similarity(old_features, features)
                loss += self._weight_pod_loss * pod_flat_loss
                self._metrics["flat"] += pod_flat_loss.item()

            if self._pod_spatial_config:
                if self._pod_spatial_config.get("scheduled_factor", False):
                    factor = self._pod_spatial_config["scheduled_factor"] * math.sqrt(
                        self._n_classes / self._task_size
                    )
                else:
                    factor = self._pod_spatial_config.get("factor", 1.)

                try:
                    pod_spatial_loss = factor * losses.pod(old_atts,atts,memory_flags=memory_flags.bool(),task_percent=(self._task + 1) / self._n_tasks,**self._pod_spatial_config)
                except:
                    import pdb
                    pdb.set_trace()
                loss += self._weight_pod_loss * pod_spatial_loss
                self._metrics["pod"] += pod_spatial_loss.item()

            if self._perceptual_features:
                percep_feat = losses.perceptual_features_reconstruction(
                    old_atts, atts, **self._perceptual_features
                )
                loss += percep_feat
                self._metrics["p_feat"] += percep_feat.item()

            if self._perceptual_style:
                percep_style = losses.perceptual_style_reconstruction(
                    old_atts, atts, **self._perceptual_style
                )
                loss += percep_style
                self._metrics["p_sty"] += percep_style.item()

            if self._gradcam_distil:
                top_logits_indexes = logits[..., :-self._task_size].argmax(dim=1)
                try:
                    onehot_top_logits = utils.to_onehot(
                        top_logits_indexes, self._n_classes - self._task_size
                    ).to(self._device)
                except:
                    import pdb
                    pbd.set_trace()

                old_logits = old_outputs["logits"]

                logits[
                    ..., :-self._task_size].backward(gradient=onehot_top_logits, retain_graph=True)
                old_logits.backward(gradient=onehot_top_logits)

                if len(outputs["gradcam_gradients"]) > 1:
                    gradcam_gradients = torch.cat(
                        [g.to(self._device) for g in outputs["gradcam_gradients"] if g is not None]
                    )
                    gradcam_activations = torch.cat(
                        [
                            a.to(self._device)
                            for a in outputs["gradcam_activations"]
                            if a is not None
                        ]
                    )
                else:
                    gradcam_gradients = outputs["gradcam_gradients"][0]
                    gradcam_activations = outputs["gradcam_activations"][0]

                if self._gradcam_distil.get("scheduled_factor", False):
                    factor = self._gradcam_distil["scheduled_factor"] * math.sqrt(
                        self._n_classes / self._task_size
                    )
                else:
                    factor = self._gradcam_distil.get("factor", 1.)

                try:
                    attention_loss = factor * losses.gradcam_distillation(
                        gradcam_gradients, old_outputs["gradcam_gradients"][0].detach(),
                        gradcam_activations, old_outputs["gradcam_activations"][0].detach()
                    )
                except:
                    import pdb
                    pdb.set_trace()

                self._metrics["grad"] += attention_loss.item()
                loss += attention_loss

                self._old_model.zero_grad()
                self._network.zero_grad()

        return loss


class BoundClipper:

    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __call__(self, module):
        if hasattr(module, "mtl_weight"):
            module.mtl_weight.data.clamp_(min=self.lower_bound, max=self.upper_bound)
        if hasattr(module, "mtl_bias"):
            module.mtl_bias.data.clamp_(min=self.lower_bound, max=self.upper_bound)
