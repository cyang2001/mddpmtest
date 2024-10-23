import os
import warnings
import torch
import hydra
import pickle
import wandb
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, open_dict
from typing import List, Optional
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import LightningLoggerBase, WandbLogger
from src.utils import utils

torch.multiprocessing.set_sharing_strategy('file_system')
os.environ['NUMEXPR_MAX_THREADS'] = '16'
warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)

log = utils.get_logger(__name__)  # init logger

@hydra.main(config_path='configs', config_name='config')  # Hydra decorator
def train(cfg: DictConfig) -> Optional[float]:
    results = {}

    # base names for logging
    base = cfg.callbacks.model_checkpoint.monitor  # naming of logs
    if 'early_stop' in cfg.callbacks:
        base_es = cfg.callbacks.early_stop.monitor  # early stop base metric

    # Set plugins for lightning trainer
    if cfg.trainer.get('accelerator', None) == 'ddp':  # for better performance in ddp mode
        plugs = DDPPlugin(find_unused_parameters=False)
    else:
        plugs = None

    if "seed" in cfg:  # for deterministic training (covers pytorch, numpy and python.random)
        log.info(f"Seed specified to {cfg.seed} by config")
        seed_everything(cfg.seed, workers=True)

    # get start and end fold
    start_fold = cfg.get('start_fold', 0)
    end_fold = cfg.get('num_folds', 5)
    if start_fold != 0:
        log.info(f'Skipping the first {start_fold} fold(s)')

    # iterate over folds from start_fold to num_fold
    for fold in range(start_fold, end_fold):  # iterate over folds

        log.info(f"Training Fold {fold+1} of {end_fold}")
        prefix = f'{fold+1}/'  # naming of logs

        # Set datamodule target
        cfg.datamodule._target_ = f'src.datamodules.Datamodules_train.{cfg.datamodule.cfg.name}'
        log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
        datamodule_train: LightningDataModule = hydra.utils.instantiate(cfg.datamodule, fold=fold)

        # Init lightning model
        log.info(f"Instantiating model <{cfg.model._target_}>")
        model: LightningModule = hydra.utils.instantiate(cfg.model, prefix=prefix)

        # Setup callbacks
        cfg.callbacks.model_checkpoint.monitor = f'{prefix}' + base  # naming of logs for cross validation
        cfg.callbacks.model_checkpoint.filename = "epoch-{epoch}_step-{step}_loss-{" + f"{prefix}" + "val/loss:.2f}"  # naming of logs

        if 'early_stop' in cfg.callbacks:
            cfg.callbacks.early_stop.monitor = f'{prefix}' + base_es

        if 'log_image_predictions' in cfg.callbacks:
            cfg.callbacks.log_image_predictions.prefix = prefix

        # Init callbacks
        callbacks: List[Callback] = []
        if "callbacks" in cfg:
            for _, cb_conf in cfg.callbacks.items():
                if "_target_" in cb_conf:
                    if cb_conf._target_ == 'pytorch_lightning.callbacks.ModelCheckpoint':
                        # Skip default ModelCheckpoint, use custom one
                        continue
                    log.info(f"Instantiating callback <{cb_conf._target_}>")
                    callbacks.append(hydra.utils.instantiate(cb_conf))

        # Add custom checkpoint callback
        custom_checkpoint_callback = CustomModelCheckpoint(
            save_dir='./checkpoints',
            save_every_n_epochs=2,
            monitor=None,
            save_last=False,
            save_top_k=0,
            verbose=True,
        )
        callbacks.append(custom_checkpoint_callback)

        # Init lightning loggers
        logger: List[LightningLoggerBase] = []
        if "logger" in cfg:
            for _, lg_conf in cfg.logger.items():
                if "_target_" in lg_conf:
                    # Set WandB id and resume parameters
                    if lg_conf._target_ == 'pytorch_lightning.loggers.WandbLogger':
                        lg_conf.id = cfg.get('wandb_id', None)  # WandB will generate a new id if None
                        lg_conf.resume = cfg.get('wandb_resume', False)
                        lg_conf.group = cfg.name  # set group name
                    log.info(f"Instantiating logger <{lg_conf._target_}>")
                    logger.append(hydra.utils.instantiate(lg_conf))

        # Get WandB logger instance
        wandb_logger = None
        for lg in logger:
            if isinstance(lg, WandbLogger):
                wandb_logger = lg
                break

        if wandb_logger is not None:
            # If not resuming, get new wandb_id
            if not cfg.get('resume', False):
                cfg.wandb_id = wandb_logger.experiment.id
                log.info(f"Generated new WandB run ID: {cfg.wandb_id}")
        else:
            log.warning("WandB logger not found.")

        # Check if need to resume training
        if cfg.get('resume', False):
            resume_ckpt = cfg.get('resume_from_checkpoint', 'checkpoints/model.ckpt')
            # Use get_original_cwd() to construct absolute path
            resume_ckpt = os.path.join(get_original_cwd(), resume_ckpt)
            if os.path.exists(resume_ckpt):
                log.info(f"Resuming training from checkpoint: {resume_ckpt}")
                # Load WandB id from checkpoint
                wandb_id = load_run_id_from_checkpoint(resume_ckpt)
                cfg.wandb_id = wandb_id
                cfg.wandb_resume = 'must'
                cfg.resume_from_checkpoint = resume_ckpt  # Update path in cfg
            else:
                log.error("Checkpoint path does not exist. Starting training from scratch.")
                cfg.wandb_id = None
                cfg.wandb_resume = False
                cfg.resume_from_checkpoint = None  # Ensure not using checkpoint
        else:
            cfg.wandb_id = None
            cfg.wandb_resume = False
            cfg.resume_from_checkpoint = None  # Ensure not using checkpoint

        # Initialize trainer
        trainer_args = {
            'callbacks': callbacks,
            'logger': logger,
            '_convert_': "partial",
            'plugins': plugs,
        }
        if cfg.get('resume', False):
            trainer_args['resume_from_checkpoint'] = cfg.resume_from_checkpoint

        trainer: Trainer = hydra.utils.instantiate(cfg.trainer, **trainer_args)

        # Log hyperparameters
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(
            config=cfg,
            model=model,
            datamodule=datamodule_train,
            trainer=trainer,
            callbacks=callbacks,
            logger=logger,
        )

        # Training
        if cfg.get('train', True):
            trainer.fit(model, datamodule_train)
            validation_metrics = trainer.callback_metrics
        else:
            # Load model state dict from checkpoint
            model_ckpt_path = cfg.get('model_checkpoint_path', 'path/to/your/checkpoint')
            model.load_state_dict(torch.load(model_ckpt_path)['state_dict'])

        # Logging
        if trainer.checkpoint_callback.best_model_path:
            log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")
            log.info(f"Best checkpoint metric:\n{trainer.checkpoint_callback.best_model_score}")
            if wandb_logger:
                wandb_logger.log_metrics({'best_ckpt_path': trainer.checkpoint_callback.best_model_path})
                wandb_logger.log_metrics({'logdir': trainer.log_dir})

        # Metrics
        validation_metrics = trainer.callback_metrics
        for key in validation_metrics:
            key = key[2:]
            valkey = prefix + key
            if 'train' not in key and 'test' not in key:
                if key not in results:
                    results[key] = []
                results[key].append(validation_metrics[valkey])

        # Evaluation
        if cfg.get("test_after_training", False):
            log.info(f"Starting evaluation phase of fold {fold+1}!")
            preds_dict = {'val': {}, 'test': {}}

            # Define test sets
            sets = {
                't2': ['Datamodules_eval.Brats21', 'Datamodules_eval.MSLUB', 'Datamodules_train.IXI'],
            }

            for dataset_name in cfg.datamodule.cfg.testsets:
                if dataset_name not in sets[cfg.datamodule.cfg.mode]:
                    continue

                cfg.datamodule._target_ = f'src.datamodules.{dataset_name}'
                log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
                datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule, fold=fold)
                datamodule.setup()

                # Validation
                log.info(f"Validation of {dataset_name}!")
                ckpt_path = cfg.get('ckpt_path', None)

                if 'train' in dataset_name:
                    trainer.test(model=model, dataloaders=datamodule.val_eval_dataloader(), ckpt_path=ckpt_path)
                else:
                    trainer.test(model=model, dataloaders=datamodule.val_dataloader(), ckpt_path=ckpt_path)

                # Collect results
                preds_dict['val'][dataset_name] = trainer.lightning_module.eval_dict
                log_dict = utils.summarize(preds_dict['val'][dataset_name], 'val')

                # Test
                log.info(f"Test of {dataset_name}!")
                if 'train' in dataset_name:
                    trainer.test(model=model, dataloaders=datamodule.test_eval_dataloader(), ckpt_path=ckpt_path)
                else:
                    trainer.test(model=model, dataloaders=datamodule.test_dataloader(), ckpt_path=ckpt_path)

                preds_dict['test'][dataset_name] = trainer.lightning_module.eval_dict
                log_dict.update(utils.summarize(preds_dict['test'][dataset_name], 'test'))
                log_dict = utils.summarize(log_dict, f'{fold+1}/' + dataset_name)
                if wandb_logger:
                    wandb_logger.log_metrics(log_dict)

            # Save predictions
            if cfg.get('pickle_preds', True):
                with open(os.path.join(trainer.log_dir, f'{fold+1}_preds_dict.pkl'), 'wb') as f:
                    pickle.dump(preds_dict, f)

    # Finalize
    log.info("Finalizing!")
    utils.finish(
        config=cfg,
        model=model,
        datamodule=datamodule_train,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

# Custom callback class

class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, save_dir='checkpoints', save_every_n_epochs=2, *args, **kwargs):
        # Get original working directory
        original_cwd = get_original_cwd()
        # Set save_dir relative to original cwd
        save_dir = os.path.join(original_cwd, save_dir)
        super().__init__(*args, **kwargs)
        self.save_dir = save_dir
        self.save_every_n_epochs = save_every_n_epochs
        os.makedirs(self.save_dir, exist_ok=True)

    def on_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if (epoch + 1) % self.save_every_n_epochs == 0:
            # Build model file path
            filepath = os.path.join(self.save_dir, 'model.ckpt')
            # Remove existing file if exists
            if os.path.exists(filepath):
                os.remove(filepath)
            # Get WandB run ID
            wandb_id = None
            wandb_experiment = None
            if isinstance(pl_module.logger, WandbLogger):
                wandb_experiment = pl_module.logger.experiment
                wandb_id = wandb_experiment.id
            elif isinstance(pl_module.logger, list):
                for logger in pl_module.logger:
                    if isinstance(logger, WandbLogger):
                        wandb_experiment = logger.experiment
                        wandb_id = wandb_experiment.id
                        break
            if wandb_id is None:
                # Log warning if WandB ID not found
                log.warning("WandB run ID not found. Checkpoint will not contain wandb_id.")
            # Save checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': pl_module.state_dict(),
                'optimizer_state_dict': trainer.optimizers[0].state_dict(),
                'lr_scheduler_state_dict': trainer.lr_schedulers[0]['scheduler'].state_dict() if trainer.lr_schedulers else None,
                'wandb_id': wandb_id  # Save WandB ID
            }
            torch.save(checkpoint, filepath)
            # Log message
            message = f"Checkpoint saved at epoch {epoch+1} to {filepath}"
            if wandb_experiment is not None:
                wandb_experiment.log({"message": message})
            else:
                log.info(message)

def load_run_id_from_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    wandb_id = checkpoint.get('wandb_id', None)
    if wandb_id is None:
        raise ValueError("WandB run ID not found in the checkpoint.")
    return wandb_id