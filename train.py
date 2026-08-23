"""Main training entry point for pre-training and downstream fine-tuning.

"""

import json
import os
import random
import time
from functools import wraps
from typing import Callable, List, Sequence

import fsspec
import hydra
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn

import src.models.nn.utils as U
import src.utils as utils
import src.utils.train
from src.dataloaders import SequenceDataset  # TODO make registry
from src.tasks import decoders, encoders, tasks
from src.utils import registry
from src.utils.optim_groups import add_optimizer_hooks

log = src.utils.train.get_logger(__name__)

# Turn on TensorFloat32 (speeds up large model training substantially)
import torch.backends

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

OmegaConf.register_new_resolver('eval', eval)
OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)
OmegaConf.register_new_resolver('min', lambda x, y: min([x, y]))


# Lots of annoying hacks to get WandbLogger to continuously retry on failure
class DummyExperiment:
    """Dummy experiment."""

    def nop(self, *args, **kw):
        pass

    def __getattr__(self, _):
        return self.nop

    def __getitem__(self, idx) -> "DummyExperiment":
        # enables self.logger.experiment[0].add_image(...)
        return self

    def __setitem__(self, *args, **kwargs) -> None:
        pass


def rank_zero_experiment(fn: Callable) -> Callable:
    """Returns the real experiment on rank 0 and otherwise the DummyExperiment."""

    @wraps(fn)
    def experiment(self):
        @rank_zero_only
        def get_experiment():
            return fn(self)

        return get_experiment() or DummyExperiment()

    return experiment


class CustomWandbLogger(WandbLogger):

    def __init__(self, *args, **kwargs):
        """Modified logger that insists on a wandb.init() call and catches wandb's error if thrown."""

        super().__init__(*args, **kwargs)

    @property
    @rank_zero_experiment
    def experiment(self):
        r"""
        Actual wandb object. To use wandb features in your
        :class:`~pytorch_lightning.core.lightning.LightningModule` do the following.
        Example::
            code-block:: python
            self.logger.experiment.some_wandb_function()
        """
        if self._experiment is None:
            if self._offline:
                os.environ["WANDB_MODE"] = "dryrun"

            attach_id = getattr(self, "_attach_id", None)
            if wandb.run is not None:
                # wandb process already created in this instance
                rank_zero_warn(
                    "There is a wandb run already in progress and newly created instances of `WandbLogger` will reuse"
                    " this run. If this is not desired, call `wandb.finish()` before instantiating `WandbLogger`."
                )
                self._experiment = wandb.run
            elif attach_id is not None and hasattr(wandb, "_attach"):
                # attach to wandb process referenced
                self._experiment = wandb._attach(attach_id)
            else:
                # create new wandb process
                while True:
                    try:
                        self._experiment = wandb.init(**self._wandb_init)
                        break
                    except Exception as e:
                        log.error("wandb Exception:\n", e)
                        t = random.randint(30, 60)
                        log.warning(f"Sleeping for {t} seconds")
                        time.sleep(t)

                # define default x-axis
                if getattr(self._experiment, "define_metric", None):
                    self._experiment.define_metric("trainer/global_step")
                    self._experiment.define_metric("*", step_metric="trainer/global_step", step_sync=True)

        return self._experiment


class SequenceLightningModule(pl.LightningModule):
    def __init__(self, config):
        # Disable profiling executor. This reduces memory and increases speed.
        try:
            torch._C._jit_set_profiling_executor(False)
            torch._C._jit_set_profiling_mode(False)
        except AttributeError:
            pass

        super().__init__()
        # Passing in config expands it one level: access by self.hparams.train instead of self.hparams.config.train
        self.save_hyperparameters(config, logger=False)

        # Dataset arguments
        self.dataset = SequenceDataset.registry[self.hparams.dataset._name_](
            **self.hparams.dataset
        )

        # Check hparams
        self._check_config()

        # PL has some bugs, so add hooks and make sure they're only called once
        self._has_setup = False

        # To be set in `setup`
        self.encoder, self.decoder, self.model = None, None, None
        self.task, self.loss, self.loss_val = None, None, None
        self.metrics, self.train_torchmetrics, self.val_torchmetrics, self.test_torchmetrics = None, None, None, None
        self.setup()

        self._state = None
        self.val_loader_names, self.test_loader_names = None, None
  


    def setup(self, stage=None):
        # We need to set up the model in setup() because for some reason when training with DDP, one GPU uses much more
        # memory than the others.
        # In order to not overwrite the model multiple times during different stages, we need this hack
        # TODO PL 1.5 seems to have an option to skip hooks to avoid this
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/5410#issuecomment-762257024
        if self._has_setup:
            return
        else:
            self._has_setup = True

        if not self.hparams.train.disable_dataset:
            self.dataset.setup()

        # Convenience feature: if model specifies encoder, combine it with main encoder
        encoder_cfg = utils.to_list(self.hparams.encoder) + utils.to_list(
            self.hparams.model.pop("encoder", None)
        )
        decoder_cfg = utils.to_list(
            self.hparams.model.pop("decoder", None)
        ) + utils.to_list(self.hparams.decoder)

        # Instantiate model
        config_path = self.hparams.model.pop("config_path", None)
        if config_path is not None:
            with open(config_path) as f:
                model_config_from_file = json.load(f)
            self.hparams.model.update(model_config_from_file)
            # Check if dropout_layer_norm is compiled
            try:
                from flash_attn.ops.layer_norm import dropout_add_layer_norm
            except ImportError:
                if self.hparams.model.get("fused_dropout_add_ln", None) is not None:
                    self.hparams.model.update({"fused_dropout_add_ln": False})
        # TODO: Hacky way to get complement_map for Caduceus models; need to find a more elegant implementation
        if "caduceus" in self.hparams.model.get("_name_"):
            OmegaConf.update(
                self.hparams.model.config, "complement_map", self.dataset.tokenizer.complement_map, force_add=True
            )
        # Instantiate the config class if using hydra's _target_ paradigm for the config
        if self.hparams.model.get("config", None) is not None and self.hparams.model.config.get("_target_", None) is not None:
            model_hparams = OmegaConf.to_container(self.hparams.model, resolve=True)
            model_hparams["config"] = hydra.utils.instantiate(model_hparams["config"])
            self.model = utils.instantiate(registry.model, model_hparams)
        else:
            self.model = utils.instantiate(registry.model, self.hparams.model)
        if (name := self.hparams.train.post_init_hook['_name_']) is not None:
            kwargs = self.hparams.train.post_init_hook.copy()
            del kwargs['_name_']
            for module in self.modules():
                if hasattr(module, name):
                    getattr(module, name)(**kwargs)

        # if self.hparams.train.get("compile_model", False):
        #     self.model = torch.compile(self.model, dynamic=False)

        # Instantiate the task
        self.task = utils.instantiate(
            tasks.registry, self.hparams.task, dataset=self.dataset, model=self.model
        )

        # Create encoders and decoders
        encoder = encoders.instantiate(
            encoder_cfg, dataset=self.dataset, model=self.model
        )
        decoder = decoders.instantiate(
            decoder_cfg, model=self.model, dataset=self.dataset
        )

        # Extract the modules, so they show up in the top level parameter count
        self.encoder = U.PassthroughSequential(self.task.encoder, encoder)
        self.decoder = U.PassthroughSequential(decoder, self.task.decoder)
        self.loss = self.task.loss
        self.loss_val = self.task.loss
        if hasattr(self.task, 'loss_val'):
            self.loss_val = self.task.loss_val
        self.metrics = self.task.metrics
        self.train_torchmetrics = self.task.train_torchmetrics
        self.val_torchmetrics = self.task.val_torchmetrics
        self.test_torchmetrics = self.task.test_torchmetrics

    def load_state_dict(self, state_dict, strict=False):
        log.info("Custom load_state_dict function is running.")

        # Exact Lightning resume/test loading must restore the complete module,
        # including downstream encoders and decoders.  Pretrained-model hooks
        # deliberately discard task-specific heads, so they are applied only
        # in load_compatible_pretrained() below and never in this generic
        # state-dict entry point.
        return super().load_state_dict(state_dict, strict=strict)

    def load_compatible_pretrained(self, checkpoint_path):
        """Warm-start from keys whose names and shapes match this model.

        This is intentionally separate from Lightning resume. Resume must use
        ``train.ckpt`` and restores the exact model, optimizer, scheduler, and
        global step. This method initializes a new architecture from a baseline
        checkpoint and never silently accepts shape mismatches.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        source_state = checkpoint.get("state_dict", checkpoint)

        hook_config = self.hparams.train.get("pretrained_model_state_hook", None)
        if hook_config is not None and hook_config.get("_name_", None) is not None:
            model_state_hook = utils.instantiate(
                registry.model_state_hook,
                hook_config.copy(),
                partial=True,
            )
            # Hooks such as load_backbone are warm-start transformations: they
            # map an MLM checkpoint onto a downstream backbone and intentionally
            # leave the new task head at its initialization.
            source_state = model_state_hook(self.model, dict(source_state))

        current_state = self.state_dict()

        compatible = {}
        unexpected = []
        shape_mismatches = {}
        for name, value in source_state.items():
            if name not in current_state:
                unexpected.append(name)
            elif current_state[name].shape != value.shape:
                shape_mismatches[name] = {
                    "checkpoint": list(value.shape),
                    "model": list(current_state[name].shape),
                }
            else:
                compatible[name] = value

        missing = sorted(set(current_state) - set(compatible))
        result = self.load_state_dict(compatible, strict=False)

        backbone_keys = [
            name
            for name in current_state
            if name.startswith("model.") and "memory_" not in name
        ]
        loaded_backbone = sum(
            current_state[name].numel() for name in backbone_keys if name in compatible
        )
        total_backbone = sum(current_state[name].numel() for name in backbone_keys)
        backbone_ratio = loaded_backbone / max(1, total_backbone)

        report = {
            "checkpoint": str(checkpoint_path),
            "loaded_keys": len(compatible),
            "missing_keys": missing,
            "unexpected_keys": sorted(unexpected),
            "shape_mismatches": shape_mismatches,
            "backbone_parameter_ratio": backbone_ratio,
        }
        log.info("Compatible warm-start report:\n%s", json.dumps(report, indent=2))
        return result, report

    def _check_config(self):
        assert self.hparams.train.state.mode in [None, "none", "null", "reset", "bptt", "tbptt"]
        assert (
                (n := self.hparams.train.state.n_context) is None
                or isinstance(n, int)
                and n >= 0
        )
        assert (
                (n := self.hparams.train.state.n_context_eval) is None
                or isinstance(n, int)
                and n >= 0
        )

    def _initialize_state(self):
        """Called at model setup and start of epoch to completely reset state"""
        self._state = None
        self._memory_chunks = []

    def _reset_state(self, batch, device=None):
        """Called to construct default_state when necessary, e.g. during BPTT"""
        device = device or batch[0].device
        self._state = self.model.default_state(*batch[0].shape[:1], device=device)

    def _detach_state(self, state):
        if isinstance(state, torch.Tensor):
            return state.detach()
        elif isinstance(state, tuple):
            return tuple(self._detach_state(s) for s in state)
        elif isinstance(state, list):
            return [self._detach_state(s) for s in state]
        elif isinstance(state, dict):
            return {k: self._detach_state(v) for k, v in state.items()}
        elif state is None:
            return None
        else:
            raise NotImplementedError

    def _process_state(self, batch, batch_idx, training=True):
        """Handle logic for state context."""
        # Number of context steps
        key = "n_context" if training else "n_context_eval"
        n_context = self.hparams.train.state.get(key)

        # Don't need to do anything if 0 context steps. Make sure there is no state
        if n_context == 0 and self.hparams.train.state.mode not in ['tbptt']:
            self._initialize_state()
            return

        # Reset state if needed
        if self.hparams.train.state.mode == "reset":
            if batch_idx % (n_context + 1) == 0:
                self._reset_state(batch)

        # Pass through memory chunks
        elif self.hparams.train.state.mode == "bptt":
            self._reset_state(batch)
            with torch.no_grad():  # should be unnecessary because individual modules should handle this
                for _batch in self._memory_chunks:
                    self.forward(_batch)
            # Prepare for next step
            self._memory_chunks.append(batch)
            self._memory_chunks = self._memory_chunks[-n_context:]

        elif self.hparams.train.state.mode == 'tbptt':
            _, _, z = batch
            reset = z["reset"]
            if reset:
                self._reset_state(batch)
            else:
                self._state = self._detach_state(self._state)

    def forward(self, batch):
        return self.task.forward(batch, self.encoder, self.model, self.decoder, self._state)

    def step(self, x_t):
        x_t, *_ = self.encoder(x_t)  # Potential edge case for encoders that expect (B, L, H)?
        x_t, state = self.model.step(x_t, state=self._state)
        self._state = state
        x_t, *_ = self.decoder.step(x_t, state=state)
        return x_t
        

    def _shared_step(self, batch, batch_idx, prefix="train"):
        """Shared step logic between training, validation, and test"""
        self._process_state(batch, batch_idx, training=(prefix == "train"))
        x, y, w = self.forward(batch)
    
       

        # Loss
        if prefix == 'train':
            loss = self.loss(x, y, **w)
        else:
            loss = self.loss_val(x, y, **w)
        


        # Metrics
        metrics = self.metrics(x, y, **w)
        metrics["loss"] = loss
        metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

        # Calculate torchmetrics
        torchmetrics = getattr(self, f'{prefix}_torchmetrics')
        torchmetrics(x, y, loss=loss)

        log_on_step = 'eval' in self.hparams and self.hparams.eval.get('log_on_step', False) and prefix == 'train'

        self.log_dict(
            metrics,
            on_step=log_on_step,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=False,
            sync_dist=True,
        )

        # log the whole dict, otherwise lightning takes the mean to reduce it
        # https://pytorch-lightning.readthedocs.io/en/stable/visualize/logging_advanced.html#enable-metrics-for-distributed-training
        self.log_dict(
            torchmetrics,
            on_step=log_on_step,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=False,
            sync_dist=True,
        )
        return loss

    def on_train_epoch_start(self):
        # Reset training torchmetrics
        self.task._reset_torchmetrics("train")

    def training_epoch_end(self, outputs):
        # Log training torchmetrics
        super().training_epoch_end(outputs)

    def on_validation_epoch_start(self):
        # Reset all validation torchmetrics
        for name in self.val_loader_names:
            self.task._reset_torchmetrics(name)

    def validation_epoch_end(self, outputs):
        # Log all validation torchmetrics
        super().validation_epoch_end(outputs)

    def on_test_epoch_start(self):
        # Reset all test torchmetrics
        for name in self.test_loader_names:
            self.task._reset_torchmetrics(name)

    def test_epoch_end(self, outputs):
        # Log all test torchmetrics
        super().test_epoch_end(outputs)

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self._shared_step(batch, batch_idx, prefix="train")

        # Log the loss explicitly so that it shows up in WandB
        # Note that this currently runs into a bug in the progress bar with ddp (as of 1.4.6)
        # https://github.com/PyTorchLightning/pytorch-lightning/pull/9142
        # We additionally log the epochs under 'trainer' to get a consistent prefix with 'global_step'
        loss_epoch = {"trainer/loss": loss, "trainer/epoch": float(self.current_epoch)}
        self.log_dict(
            loss_epoch,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            add_dataloader_idx=False,
            sync_dist=True,
        )

        # Log any extra info that the models want to expose (e.g. output norms)
        metrics = {}
        for module in list(self.modules())[1:]:
            if hasattr(module, "metrics"):
                metrics.update(module.metrics)

        self.log_dict(
            metrics,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            add_dataloader_idx=False,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # There's a bit of an annoying edge case with the first (0-th) epoch; it has to be excluded due to the initial
        # sanity check
        ema = (
                self.val_loader_names[dataloader_idx].endswith("/ema")
                and self.optimizers().optimizer.stepped
        )
        if ema:
            self.optimizers().swap_ema()
        loss = self._shared_step(
            batch, batch_idx, prefix=self.val_loader_names[dataloader_idx]
        )
        if ema:
            self.optimizers().swap_ema()

        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self._shared_step(
            batch, batch_idx, prefix=self.test_loader_names[dataloader_idx]
        )

    def configure_optimizers(self):
        # Set zero weight decay for some params
        if 'optimizer_param_grouping' in self.hparams.train:
            add_optimizer_hooks(self.model, **self.hparams.train.optimizer_param_grouping)

        # Normal parameters
        all_params = list(self.parameters())
        params = [p for p in all_params if not hasattr(p, "_optim")]

        optimizer = utils.instantiate(registry.optimizer, self.hparams.optimizer, params)

        del self.hparams.optimizer._name_

        # Add parameters with special hyperparameters
        hps = [getattr(p, "_optim") for p in all_params if hasattr(p, "_optim")]
        hps = [
            # dict(s) for s in set(frozenset(hp.items()) for hp in hps)
            dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
            # dict(s) for s in dict.fromkeys(frozenset(hp.items()) for hp in hps)
        ]  # Unique dicts
       
        for hp in hps:
            params = [p for p in all_params if getattr(p, "_optim", None) == hp]
            optimizer.add_param_group(
                {"params": params, **self.hparams.optimizer, **hp}
            )

        # Layer Decay
        if self.hparams.train.layer_decay['_name_'] is not None:
            get_num_layer = utils.instantiate(
                registry.layer_decay,
                self.hparams.train.layer_decay['_name_'],
                partial=True,
            )

            # Go through all parameters and get num layer
            layer_wise_groups = {}
            num_max_layers = 0
            for name, p in self.named_parameters():
                # Get layer id for each parameter in the model
                layer_id = get_num_layer(name)

                # Add to layer wise group
                if layer_id not in layer_wise_groups:
                    layer_wise_groups[layer_id] = {
                        'params': [],
                        'lr': None,
                        'weight_decay': self.hparams.optimizer.weight_decay
                    }
                layer_wise_groups[layer_id]['params'].append(p)

                if layer_id > num_max_layers:
                    num_max_layers = layer_id

            # Update lr for each layer
            for layer_id, group in layer_wise_groups.items():
                group['lr'] = self.hparams.optimizer.lr * (
                        self.hparams.train.layer_decay.decay ** (num_max_layers - layer_id))

            # Reset the torch optimizers param groups
            optimizer.param_groups = []
            for layer_id, group in layer_wise_groups.items():
                optimizer.add_param_group(group)

        # Print optimizer info for debugging
        keys = set([k for hp in hps for k in hp.keys()])  # Special hparams
        utils.train.log_optimizer(log, optimizer, keys)
        # Configure scheduler
        if "scheduler" not in self.hparams:
            return optimizer
        lr_scheduler = utils.instantiate(
            registry.scheduler, self.hparams.scheduler, optimizer
        )
        scheduler = {
            "scheduler": lr_scheduler,
            "interval": self.hparams.train.interval,  # 'epoch' or 'step'
            "monitor": self.hparams.train.monitor,
            "name": "trainer/lr",  # default is e.g. 'lr-AdamW'
        }
        # See documentation for how to configure the return
        # https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.core.lightning.html#pytorch_lightning.core.lightning.LightningModule.configure_optimizers
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return self.dataset.train_dataloader(**self.hparams.loader)

    def _eval_dataloaders_names(self, loaders, prefix):
        """Process loaders into a list of names and loaders"""
        if utils.is_dict(loaders):
            return [
                f"{prefix}/{k}" if k is not None else prefix for k in loaders.keys()
            ], list(loaders.values())
        elif utils.is_list(loaders):
            return [f"{prefix}/{i}" for i in range(len(loaders))], loaders
        else:
            return [prefix], [loaders]

    def _eval_dataloaders(self):
        # Return all val + test loaders
        val_loaders = self.dataset.val_dataloader(**self.hparams.loader)
        test_loaders = self.dataset.test_dataloader(**self.hparams.loader)
        val_loader_names, val_loaders = self._eval_dataloaders_names(val_loaders, "val")
        test_loader_names, test_loaders = self._eval_dataloaders_names(
            test_loaders, "test"
        )

        # Duplicate datasets for ema
        if self.hparams.train.ema > 0.0:
            val_loader_names += [name + "/ema" for name in val_loader_names]
            val_loaders = val_loaders + val_loaders
            test_loader_names += [name + "/ema" for name in test_loader_names]
            test_loaders = test_loaders + test_loaders

        # adding option to only have val loader at eval (e.g., if test is duplicate)
        eval_loader_names = []
        eval_loaders = []
        if not self.hparams.train.get("remove_val_loader_in_eval", False):
            eval_loader_names += val_loader_names
            eval_loaders += val_loaders
        if not self.hparams.train.get("remove_test_loader_in_eval", False):
            eval_loader_names += test_loader_names
            eval_loaders += test_loaders
        return eval_loader_names, eval_loaders

    def val_dataloader(self):
            #  validation loop 只跑 val，用来选 best ckpt
            val_loaders = self.dataset.val_dataloader(**self.hparams.loader)
            val_loader_names, val_loaders = self._eval_dataloaders_names(val_loaders, "val")

            # EMA ，val 照旧复制一份
            if self.hparams.train.ema > 0.0:
                val_loader_names += [name + "/ema" for name in val_loader_names]
                val_loaders = val_loaders + val_loaders

            self.val_loader_names = val_loader_names
            return val_loaders


    def test_dataloader(self):
        # Strict final eval: only use *true* test loaders here
        test_loaders = self.dataset.test_dataloader(**self.hparams.loader)
        test_loader_names, test_loaders = self._eval_dataloaders_names(test_loaders, "test")

        # Duplicate datasets for ema
        if self.hparams.train.ema > 0.0:
            test_loader_names += [name + "/ema" for name in test_loader_names]
            test_loaders = test_loaders + test_loaders
            
        self.test_loader_names = test_loader_names
        return test_loaders




def create_trainer(config, **kwargs):
    callbacks: List[pl.Callback] = []
    logger = None

    # WandB Logging
    if config.get("wandb") is not None:
        import wandb
        logger = CustomWandbLogger(
            config=utils.to_dict(config, recursive=True),
            settings=wandb.Settings(start_method="fork"),
            **config.wandb,
        )

    # --------------------------------------------------
    # Auto-configure checkpoint saving by max_epochs
    # --------------------------------------------------
    if "callbacks" in config:
        max_epochs = config.trainer.get("max_epochs", None)
        max_steps = config.trainer.get("max_steps", None)

        if max_epochs is None and (max_steps is None or int(max_steps) <= 0):
            raise ValueError("Set either trainer.max_epochs or a positive trainer.max_steps.")

        # 周期保存
        if "periodic_checkpoint" in config.callbacks:
            if max_epochs is not None:
                save_interval = max(1, int(max_epochs) // 10)
                config.callbacks.periodic_checkpoint.every_n_epochs = save_interval
                config.callbacks.periodic_checkpoint.pop("every_n_train_steps", None)
                interval_description = f"every_n_epochs={save_interval}"
            else:
                save_interval = max(1, int(max_steps) // 10)
                config.callbacks.periodic_checkpoint.every_n_epochs = 0
                config.callbacks.periodic_checkpoint.every_n_train_steps = save_interval
                config.callbacks.periodic_checkpoint.filename = "step{step:06d}"
                interval_description = f"every_n_train_steps={save_interval}"
            config.callbacks.periodic_checkpoint.save_top_k = -1
            config.callbacks.periodic_checkpoint.save_last = False

            if config.callbacks.periodic_checkpoint.get("filename", None) is None:
                config.callbacks.periodic_checkpoint.filename = "epoch{epoch:02d}"

            log.info(
                f"[Periodic Checkpoint] max_epochs={max_epochs}, max_steps={max_steps}, "
                f"{interval_description}, save_top_k=-1, "
                f"dirpath={config.callbacks.periodic_checkpoint.get('dirpath', None)}"
            )

        # best checkpoint：保留原逻辑
        if "model_checkpoint" in config.callbacks:
            if config.callbacks.model_checkpoint.get("save_top_k", None) is None:
                config.callbacks.model_checkpoint.save_top_k = 1

            log.info(
                f"[Best Checkpoint] monitor={config.callbacks.model_checkpoint.get('monitor', None)}, "
                f"mode={config.callbacks.model_checkpoint.get('mode', None)}, "
                f"dirpath={config.callbacks.model_checkpoint.get('dirpath', None)}"
            )

    # Lightning callbacks
    if "callbacks" in config:
        for _name_, callback in config.callbacks.items():
            if config.get("wandb") is None and _name_ in ["learning_rate_monitor"]:
                continue

            # 对未注册的 periodic_checkpoint，直接按 ModelCheckpoint 实例化
            if _name_ == "periodic_checkpoint":
                log.info("Instantiating callback <pytorch_lightning.callbacks.ModelCheckpoint> for periodic_checkpoint")
                callbacks.append(
                    pl.callbacks.ModelCheckpoint(**OmegaConf.to_container(callback, resolve=True))
                )
                continue

            log.info(f"Instantiating callback <{registry.callbacks[_name_]}>")
            callback._name_ = _name_
            callbacks.append(utils.instantiate(registry.callbacks, callback))

    # Add ProgressiveResizing callback
    if "callbacks" in config and config.callbacks.get("progressive_resizing", None) is not None:
        num_stages = len(config.callbacks.progressive_resizing.stage_params)
        log.info(f"Progressive Resizing: {num_stages} stages")
        for i, e in enumerate(config.callbacks.progressive_resizing.stage_params):
            log.info(f"\tStage {i}: {e['resolution']} @ {e['epochs']} epochs")

    # Configure ddp automatically
    n_devices = config.trainer.get("devices", 1)
    if isinstance(n_devices, Sequence):
        n_devices = len(n_devices)
    if n_devices > 1 and config.trainer.get("strategy", None) is None:
        config.trainer.strategy = dict(
            _target_="pytorch_lightning.strategies.DDPStrategy",
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
        )

    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer = hydra.utils.instantiate(config.trainer, callbacks=callbacks, logger=logger)

    return trainer


def fsspec_exists(filename):
    if filename is None:
        return False
    fs, _ = fsspec.core.url_to_fs(filename)
    return fs.exists(filename)


def train(config):
    if config.train.seed is not None:
        pl.seed_everything(config.train.seed, workers=True)

    trainer = create_trainer(config)
    model = SequenceLightningModule(config)

    if config.train.get("eval_only", False):
        checkpoint_path = config.train.get("ckpt", None)
        if checkpoint_path is None or not fsspec_exists(checkpoint_path):
            raise FileNotFoundError(
                "train.eval_only=true requires train.ckpt to reference a complete checkpoint"
            )
        if config.train.get("pretrained_model_path", None) is not None:
            raise ValueError(
                "Do not combine train.eval_only=true with train.pretrained_model_path; "
                "train.ckpt already contains the complete fine-tuned module."
            )
        log.info(f"Running TEST-only evaluation using ckpt: {checkpoint_path}")
        trainer.test(model, ckpt_path=checkpoint_path)
        return

    # A pretrained model is an initialization source, not an exact resume.
    # Exact resume is handled by trainer.fit(..., ckpt_path=train.ckpt).
    if config.train.get("pretrained_model_path", None) is not None:
        if config.train.pretrained_model_strict_load:
            checkpoint = torch.load(
                config.train.pretrained_model_path,
                map_location="cpu",
            )
            state_dict = checkpoint.get("state_dict", checkpoint)
            model.load_state_dict(state_dict, strict=True)
        else:
            model.load_compatible_pretrained(config.train.pretrained_model_path)

    # Optional validation before training
    if config.train.validate_at_start:
        log.info("Running validation before training")
        trainer.validate(model)

    log.info(f"{config.train.ckpt=} {fsspec_exists(config.train.ckpt)=}")

    # Resume from last checkpoint if it exists
    if config.train.ckpt is not None and fsspec_exists(config.train.ckpt):
        trainer.fit(model, ckpt_path=config.train.ckpt)
    else:
        trainer.fit(model)

    if config.train.test:
        # test 阶段只跑 true test，不混 val
        config.train.update({"remove_val_loader_in_eval": True})
        config.train.update({"remove_test_loader_in_eval": False})

        if config.train.get("cross_validation", False):
            best_val_ckpt = os.path.join(
                model.hparams.callbacks.model_checkpoint.dirpath,
                f"{model.hparams.callbacks.model_checkpoint.filename}.ckpt",
            )

            if fsspec_exists(best_val_ckpt):
                log.info(f"Running final TEST using ckpt: {best_val_ckpt}")
                trainer.test(model, ckpt_path=best_val_ckpt)
            else:
                log.warning(
                    f"Cross-validation best checkpoint not found: {best_val_ckpt}. "
                    "Falling back to current in-memory weights for final test."
                )
                trainer.test(model)
        else:
            # Lightning 里可能有多个 ModelCheckpoint callback，手动找 monitor=val/loss 的那个
            best_ckpt_path = None
            for cb in trainer.callbacks:
                if isinstance(cb, pl.callbacks.ModelCheckpoint):
                    monitor_name = getattr(cb, "monitor", None)
                    candidate = getattr(cb, "best_model_path", None)
                    if monitor_name == config.train.monitor and candidate:
                        best_ckpt_path = candidate
                        break

            if best_ckpt_path and fsspec_exists(best_ckpt_path):
                log.info(f"Running final TEST using best checkpoint: {best_ckpt_path}")
                trainer.test(model, ckpt_path=best_ckpt_path)
            else:
                log.warning(
                    "No best checkpoint found for final TEST. "
                    "This usually means validation has not completed yet. "
                    "Falling back to current in-memory weights for final test."
                )
                trainer.test(model)



@hydra.main(config_path="configs", config_name="config.yaml")
def main(config: OmegaConf):
    # Process config:
    # - register evaluation resolver
    # - filter out keys used only for interpolation
    # - optional hooks, including disabling python warnings or debug friendly configuration
    config = utils.train.process_config(config)
    # if config.train.get("compile_model", False):
    #     # See: https://github.com/arogozhnikov/einops/wiki/Using-torch.compile-with-einops
    #     from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1
    #     allow_ops_in_compiled_graph()

    # Pretty print config using Rich library
    utils.train.print_config(config, resolve=True)

    train(config)


if __name__ == "__main__":
    main()
