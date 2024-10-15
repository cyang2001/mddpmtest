from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning.plugins import DDPPlugin
import hydra
from omegaconf import DictConfig
from typing import List, Optional
import wandb 
import os
import warnings
import torch
from src.utils import utils
from pytorch_lightning.loggers import LightningLoggerBase
import pickle

from pytorch_lightning.callbacks import ModelCheckpoint

torch.multiprocessing.set_sharing_strategy('file_system')
os.environ['NUMEXPR_MAX_THREADS'] = '16'
warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)


log = utils.get_logger(__name__) # init logger

@hydra.main(config_path='configs', config_name='config') # Hydra decorator
def train(cfg: DictConfig) -> Optional[float]: 
    results = {}

    # base names for logging
    base = cfg.callbacks.model_checkpoint.monitor # naming of logs
    if 'early_stop' in cfg.callbacks:
        base_es = cfg.callbacks.early_stop.monitor # early stop base metric

    # load checkpoint if specified
    if cfg.get('load_checkpoint')  and (cfg.get('onlyEval') or cfg.get('resume_train')): # load stored checkpoint for testing or resuming training
        wandbID, checkpoints = utils.get_checkpoint(cfg, cfg.get('load_checkpoint')) # outputs a Dictionary of checkpoints and the corresponding wandb ID to resume the run 
        if cfg.get('new_wandb_run',False): # If we want to onlyEvaluate a run in a new wandb run
            cfg.logger.wandb.id = wandb.util.generate_id()
        else:
            if cfg.get('resume_wandb',True):
                log.info(f"Resuming wandb run")
                cfg.logger.wandb.resume = wandbID # this will allow resuming the wandb Run 

    cfg.logger.wandb.group = cfg.name  # specify group name in wandb 

    # Set plugins for lightning trainer
    if cfg.trainer.get('accelerator',None) == 'ddp': # for better performance in ddp mode
        plugs = DDPPlugin(find_unused_parameters=False)
    else: 
        plugs = None

    if "seed" in cfg: # for deterministic training (covers pytorch, numpy and python.random)
        log.info(f"Seed specified to {cfg.seed} by config")
        seed_everything(cfg.seed, workers=True)

    # get start and end fold
    start_fold = cfg.get('start_fold',0)
    end_fold = cfg.get('num_folds',5)
    if start_fold != 0:
        log.info(f'skipping the first {start_fold} fold(s)') 

    # iterate over folds from start_fold to num_fold
    for fold in range(start_fold,end_fold): # iterate over folds 
        
        log.info(f"Training Fold {fold+1} of {end_fold} in the WandB group {cfg.logger.wandb.group}")
        prefix = f'{fold+1}/' # naming of logs


        cfg.datamodule._target_ = f'src.datamodules.Datamodules_train.{cfg.datamodule.cfg.name}' # set datamodule target
        log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>") 
        datamodule_train: LightningDataModule = hydra.utils.instantiate(cfg.datamodule,fold=fold) # instantiate datamodule

        # Init lightning model
        log.info(f"Instantiating model <{cfg.model._target_}>")
        model: LightningModule = hydra.utils.instantiate(cfg.model,prefix=prefix) # instantiate model

        # setup callbacks
        cfg.callbacks.model_checkpoint.monitor = f'{prefix}' + base # naming of logs for cross validation
        cfg.callbacks.model_checkpoint.filename = "epoch-{epoch}_step-{step}_loss-{"+f"{prefix}"+"val/loss:.2f}" # naming of logs for cross validation

        if 'early_stop' in cfg.callbacks:
            cfg.callbacks.early_stop.monitor = f'{prefix}' + base_es # naming of logs for cross validation

        if 'log_image_predictions' in cfg.callbacks:
            cfg.callbacks.log_image_predictions.prefix = prefix # naming of logs for cross validation
        
        # init callbacks
        callbacks: List[Callback] = []
        if "callbacks" in cfg:
            for _, cb_conf in cfg.callbacks.items():
                if "_target_" in cb_conf:
                    if cb_conf._target_ == 'pytorch_lightning.callbacks.ModelCheckpoint':
                        # 跳过默认的 ModelCheckpoint，使用自定义的
                        continue
                    log.info(f"Instantiating callback <{cb_conf._target_}>")
                    callbacks.append(hydra.utils.instantiate(cb_conf))

            # 添加自定义回调
            custom_checkpoint_callback = CustomModelCheckpoint(
                save_dir='/save', 
                save_every_n_epochs=10, 
                monitor=None,  
                save_last=False,
                save_top_k=0,  
                verbose=True,
            )
            callbacks.append(custom_checkpoint_callback)
        else:
            callbacks = []
            custom_checkpoint_callback = CustomModelCheckpoint(
                save_dir='/save',
                save_every_n_epochs=10,
                monitor=None,
                save_last=False,
                save_top_k=0,
                verbose=True,
            )
            callbacks.append(custom_checkpoint_callback)
        # Init lightning loggers
        logger: List[LightningLoggerBase] = []
        # 在实例化 logger 时，添加 resume 参数
        if "logger" in cfg:
            for _, lg_conf in cfg.logger.items():
                if "_target_" in lg_conf:
                    if lg_conf._target_ == 'pytorch_lightning.loggers.WandbLogger':
                        if cfg.get('resume', False):
                            lg_conf.resume = 'must'
                            # 从检查点加载运行 ID
                            lg_conf.id = load_run_id_from_checkpoint(cfg.resume_from_checkpoint)
                        else:
                            lg_conf.resume = 'allow'
                    log.info(f"Instantiating logger <{lg_conf._target_}>")
                    logger.append(hydra.utils.instantiate(lg_conf))

        # Load checkpoint if specified
        if cfg.get('load_checkpoint') and (cfg.get('onlyEval',False) or cfg.get('resume_train',False) ): # pass checkpoint to resume from
            with open_dict(cfg):
                cfg.trainer.resume_from_checkpoint = checkpoints[f"fold-{fold+1}"]
                cfg.ckpt_path=None
            log.info(f"Restoring Trainer State of loaded checkpoint: ",cfg.trainer.resume_from_checkpoint)
        
        # Init lightning trainer
         # 检查是否需要恢复训练
        if cfg.get('resume', False):
            resume_ckpt = cfg.get('resume_from_checkpoint', None)
            if resume_ckpt is not None and os.path.exists(resume_ckpt):
                log.info(f"Resuming training from checkpoint: {resume_ckpt}")
                # 从检查点中加载 WandB 的 id
                wandb_id = load_run_id_from_checkpoint(resume_ckpt)
                cfg.wandb_id = wandb_id
                cfg.wandb_resume = 'must'
            else:
                log.error("Checkpoint path does not exist. Starting training from scratch.")
                cfg.wandb_id = None
                cfg.wandb_resume = False
        else:
            cfg.wandb_id = None
            cfg.wandb_resume = False

        # 初始化日志记录器
        logger: List[LightningLoggerBase] = []
        if "logger" in cfg:
            for _, lg_conf in cfg.logger.items():
                if "_target_" in lg_conf:
                    # 设置 WandB 的 id 和 resume 参数
                    if lg_conf._target_ == 'pytorch_lightning.loggers.WandbLogger':
                        lg_conf.id = cfg.wandb_id
                        lg_conf.resume = cfg.wandb_resume
                    log.info(f"Instantiating logger <{lg_conf._target_}>")
                    logger.append(hydra.utils.instantiate(lg_conf))

        # 初始化训练器
        trainer: Trainer = hydra.utils.instantiate(
            cfg.trainer,
            callbacks=callbacks,
            logger=logger,
            _convert_="partial",
            plugins=plugs,
            resume_from_checkpoint=cfg.get('resume_from_checkpoint', None)
        )
        # Send some parameters from config to all lightning loggers
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(
            config=cfg,
            model=model,
            datamodule=datamodule_train,
            trainer=trainer,
            callbacks=callbacks,
            logger=logger,
        )


        if (not cfg.get('onlyEval',False) or cfg.get('resume_train',False)) : # train model
            trainer.fit(model, datamodule_train)
            validation_metrics = trainer.callback_metrics
        else: # load trained model
            model.load_state_dict(torch.load(checkpoints[f'fold-{fold+1}'])['state_dict'])

        # logging
        log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")
        log.info(f"Best checkpoint metric:\n{trainer.checkpoint_callback.best_model_score}")
        trainer.logger.experiment[0].log({'best_ckpt_path':trainer.checkpoint_callback.best_model_path})
        trainer.logger.experiment[0].log({'logdir':trainer.log_dir})

        # metrics
        validation_metrics = trainer.callback_metrics
        for key in validation_metrics:
            key =  key[2:]
            valkey= prefix + key
            if not 'train' in key and not 'test' in key:
                if key not in results:
                    results[key] = []
                results[key].append(validation_metrics[valkey])

    # Evaluate model on test set, using the best or last model from each trained fold 

        if cfg.get("test_after_training"): # and not 'simclr' in  cfg.model._target_.lower():
            log.info(f"Starting evaluation phase of fold {fold+1}!")
            preds_dict = {}
            preds_dict = {'val':{},'test':{}} # a dict for each data set
            
            sets = {
                    't2':['Datamodules_eval.Brats21','Datamodules_eval.MSLUB','Datamodules_train.IXI'],
                   }
            
                
            for set in cfg.datamodule.cfg.testsets :
                if not set in sets[cfg.datamodule.cfg.mode]: # skip testsets of different modalities
                    continue    

                cfg.datamodule._target_ = 'src.datamodules.{}'.format(set)
                log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
                datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule, fold=fold)
                datamodule.setup()

                # Validation steps
                log.info("Validation of {}!".format(set))

                ckpt_path=cfg.get('ckpt_path',None)

                if 'train' in set:
                    trainer.test(model=model,dataloaders=datamodule.val_eval_dataloader(),ckpt_path=ckpt_path)
                else: 
                    trainer.test(model=model,dataloaders=datamodule.val_dataloader(),ckpt_path=ckpt_path)
                # evaluation results
                preds_dict['val'][set] = trainer.lightning_module.eval_dict
                log_dict = utils.summarize(preds_dict['val'][set],'val') # sets prefix val/ and removes lists for better logging in wandb

                # Test steps
                log.info("Test of {}!".format(set))
                if 'train' in set:
                    trainer.test(model=model,dataloaders=datamodule.test_eval_dataloader(),ckpt_path=ckpt_path)
                else: 
                    trainer.test(model=model,dataloaders=datamodule.test_dataloader(),ckpt_path=ckpt_path)

                # log to wandb
                preds_dict['test'][set] = trainer.lightning_module.eval_dict
                log_dict.update(utils.summarize(preds_dict['test'][set],'test')) # sets prefix test/ and removes lists for better logging in wandb
                log_dict = utils.summarize(log_dict,f'{fold+1}/'+set) # sets prefix for each data set
                trainer.logger.experiment[0].log(log_dict)

                

            # pickle preds_dict for later analysis
            if cfg.get('pickle_preds',True):
                with open(os.path.join(trainer.log_dir,f'{fold+1}_preds_dict.pkl'),'wb') as f:
                    pickle.dump(preds_dict,f)



    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=cfg,
        model=model,
        datamodule=datamodule_train,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )


# 自定义回调类
class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, save_dir, save_every_n_epochs=50, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_dir = save_dir
        self.save_every_n_epochs = save_every_n_epochs
        os.makedirs(self.save_dir, exist_ok=True)

    def on_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if (epoch + 1) % self.save_every_n_epochs == 0:
            # 构建模型文件路径
            filepath = os.path.join(self.save_dir, 'model.ckpt')
            # 如果存在，则删除旧的模型文件
            if os.path.exists(filepath):
                os.remove(filepath)
            # 保存新的模型文件，包括优化器和调度器状态，以及 WandB 的 id
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': pl_module.state_dict(),
                'optimizer_state_dict': trainer.optimizers[0].state_dict(),
                'lr_scheduler_state_dict': trainer.lr_schedulers[0]['scheduler'].state_dict() if trainer.lr_schedulers else None,
                'wandb_id': pl_module.logger.experiment.id  # 保存 WandB 的 id
            }
            torch.save(checkpoint, filepath)
            pl_module.logger.info(f"Checkpoint saved at epoch {epoch+1} to {filepath}")
def load_run_id_from_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    wandb_id = checkpoint.get('wandb_id', None)
    if wandb_id is None:
        raise ValueError("WandB run ID not found in the checkpoint.")
    return wandb_id