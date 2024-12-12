from kipoi_utils.external.flatten_json import flatten
from bpnet.utils import write_json, dict_prefix_key
from keras.callbacks import EarlyStopping, CSVLogger, TensorBoard, LearningRateScheduler, ModelCheckpoint#, TerminateOnNaN
from collections import OrderedDict
import os
import gin
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class PredictandSave(ModelCheckpoint):
    def __init__(self,
                 predpath,
                 valid_data,
                 numpypath,
                 monitor='val_loss',
                 verbose=0,
                 save_best_only=True,
                 save_weights_only=True,
                 mode='min',
                 period=1):
        self.predpath = predpath
        self.valid_data = valid_data
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.numpypath = numpypath

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self._predict_on_epoch_end(epoch, logs)

    def _predict_on_epoch_end(self, epoch, logs):

        predpath = self.predpath.format(epoch=epoch + 1, **logs)
        numpypath = self.numpypath.format(epoch=epoch + 1, **logs)
        pred = self.model.predict(self.valid_data.data['inputs']['seq'], batch_size=None)
        target_key = list(self.valid_data.data['targets'].keys())[0]
        true_counts = self.valid_data.data['targets'][target_key]
        '''try:
            print(f"===================================================================type(true_counts)\n{type(true_counts)}")
        except:
            pass
        try:
            print(f"===================================================================true_counts.keys()\n{true_counts.keys()}")
        except:
            pass
        try:
            print(f"===================================================================type(pred)\n{type(pred)}")
        except:
            pass
        try:
            print(f"===================================================================len(pred)\n{len(pred)}")
        except:
            pass
        try:
            print(f"===================================================================type(pred[0])\n{type(pred[0])}")
        except:
            pass
        try:
            print(f"===================================================================len(pred[0])\n{len(pred[0])}")
        except:
            pass
        try:
            print(f"===================================================================pred[0].keys()\n{pred[0].keys()}")
        except:
            pass

        try:
            with open(numpypath, 'wb') as numpy_file:
                np.save(numpy_file, true_counts)
                np.save(numpy_file, pred[1])
        except:
            pass'''
        try:
            if target_key == 'rep1/counts':
                pred_counts = pred[1]
            else:
                pred_counts = pred[0]
            df = pd.DataFrame({"true_counts": true_counts.flatten(), "pred_counts": pred_counts.flatten()})
            df.to_csv(predpath,header=["true_counts","pred_counts"])
        except:
            pass




# Source from: https://stackoverflow.com/questions/49127214/keras-how-to-output-learning-rate-onto-tensorboard
# allows logging of learning rate on tensorboard
import keras.backend as K

class LRTensorBoard(TensorBoard):
    # add other arguments to __init__ if you need
    def __init__(self, log_dir, **kwargs):
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        lr = K.eval(self.model.optimizer.lr)
        if self.model.optimizer.initial_decay > 0:                                              
            lr = lr * (1. / (1. + self.model.optimizer.decay * K.cast(self.model.optimizer.iterations,
                                                                    K.dtype(self.model.optimizer.decay))))
        else: 
            lr = self.model.optimizer.lr

        logs.update({'lr': K.eval(lr)})
        super().on_epoch_end(epoch, logs)

@gin.configurable
class SeqModelTrainer:
    def __init__(self, model, train_dataset, valid_dataset, output_dir,
                 cometml_experiment=None, wandb_run=None):
        """
        Args:
          model: compiled keras.Model
          train: training Dataset (object inheriting from kipoi.data.Dataset)
          valid: validation Dataset (object inheriting from kipoi.data.Dataset)
          output_dir: output directory where to log the training
          cometml_experiment: if not None, append logs to commetml
        """
        # override the model class
        self.seq_model = model
        self.model = self.seq_model.model

        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.cometml_experiment = cometml_experiment
        self.wandb_run = wandb_run
        self.metrics = dict()

        if not isinstance(self.valid_dataset, list):
            # package the validation dataset into a list of validation datasets
            self.valid_dataset = [('valid', self.valid_dataset)]

        # setup the output directory
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.ckp_file = f"{self.output_dir}/model.h5"
        if os.path.exists(self.ckp_file):
            raise ValueError(f"model.h5 already exists in {self.output_dir}")
        self.history_path = f"{self.output_dir}/history.csv"
        self.evaluation_path = f"{self.output_dir}/evaluation.valid.json"
        self.checkpoint_dir = os.path.join(os.path.abspath(self.output_dir),"checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train(self,
              batch_size=256,
              epochs=100,
              early_stop_patience=4,
              num_workers=8,
              train_epoch_frac=1.0,
              valid_epoch_frac=1.0,
              train_samples_per_epoch=None,
              validation_samples=None,
              train_batch_sampler=None,
              tensorboard=True,
              lr_scheduler=None,
              min_delta=0,
              monitor='val_loss'):
        """Train the model
        Args:
          batch_size:
          epochs:
          patience: early stopping patience
          num_workers: how many workers to use in parallel
          train_epoch_frac: if smaller than 1, then make the epoch shorter
          valid_epoch_frac: same as train_epoch_frac for the validation dataset
          train_batch_sampler: batch Sampler for training. Useful for say Stratified sampling
          tensorboard: if True, tensorboard output will be added
          lr_scheduler: function that follows the requirements to pass to LearningRateScheduler
        """

        # Add debugging info
        print('\n\n')

        print("Model inputs:", self.model.inputs)
        print("Model outputs:", self.model.outputs)

        # Check variable status
        trainable_vars = self.model.trainable_weights
        print("Number of trainable variables:", len(trainable_vars))
        for var in trainable_vars:
            print(f"Variable: {var.name}, Shape: {var.shape}")

        print('\n\n')

        if train_batch_sampler is not None:
            train_it = self.train_dataset.batch_train_iter(shuffle=False,
                                                           batch_size=1,
                                                           drop_last=None,
                                                           batch_sampler=train_batch_sampler,
                                                           num_workers=num_workers)
        else:
            train_it = self.train_dataset.batch_train_iter(batch_size=batch_size,
                                                           shuffle=True,
                                                           num_workers=num_workers)
        next(train_it)
        valid_dataset = self.valid_dataset[0][1]  # take the first one
        valid_it = valid_dataset.batch_train_iter(batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=num_workers)
        next(valid_it)

        predpath = os.path.join(os.path.join(self.checkpoint_dir, "predictions_{epoch:02d}.csv"))
        numpypath = os.path.join(os.path.join(self.checkpoint_dir, "numpy_{epoch:02d}"))
        model_checkpoint_callback = PredictandSave(predpath=predpath,valid_data=self.valid_dataset[0][1],numpypath=numpypath)


        if lr_scheduler: 
            custom_callbacks = [LearningRateScheduler(lr_scheduler),model_checkpoint_callback]
        else:
            custom_callbacks = [model_checkpoint_callback]
        custom_callbacks = []
        if tensorboard:
            tb = [LRTensorBoard(log_dir=self.output_dir)]
            '''tb = [LRTensorBoard(log_dir=self.output_dir,
                                histogram_freq=1,
                                write_grads=True,
                                write_graph=True)]'''
        else:
            tb = []

        if self.wandb_run is not None:
            from wandb.keras import WandbCallback
            wcp = [WandbCallback(save_model=False)]  # we save the model using ModelCheckpoint
        else:
            wcp = []

        # train the model
        if len(valid_dataset) == 0:
            raise ValueError("len(self.valid_dataset) == 0")

        if train_samples_per_epoch is None:
            train_steps_per_epoch = max(int(len(self.train_dataset) / batch_size * train_epoch_frac), 1)
        else:
            train_steps_per_epoch = max(int(train_samples_per_epoch / batch_size), 1)

        if validation_samples is None:
            # parametrize with valid_epoch_frac
            validation_steps = max(int(len(valid_dataset) / batch_size * valid_epoch_frac), 1)
        else:
            validation_steps = max(int(validation_samples / batch_size), 1)

        self.model.fit_generator(
            train_it,
            epochs=epochs,
            steps_per_epoch=train_steps_per_epoch,
            validation_data=valid_it,
            validation_steps=validation_steps,
            callbacks=[
                EarlyStopping(
                    patience=early_stop_patience,
                    restore_best_weights=True,
                    min_delta=min_delta,
                    monitor=monitor
                ),
                CSVLogger(self.history_path)
            ] + tb + wcp + custom_callbacks
        )
        self.model.save(self.ckp_file)

        # log metrics from the best epoch
        try:
            dfh = pd.read_csv(self.history_path)
            m = dict(dfh.iloc[dfh.val_loss.idxmin()])
            if self.cometml_experiment is not None:
                self.cometml_experiment.log_metrics(m, prefix="best-epoch/")
            if self.wandb_run is not None:
                self.wandb_run.summary.update(flatten(dict_prefix_key(m, prefix="best-epoch/"), separator='/'))
        except FileNotFoundError as e:
            logger.warning(e)

    def evaluate(self, metric, batch_size=256, num_workers=8, eval_train=False, eval_skip=[], save=True, **kwargs):
        """Evaluate the model on the validation set
        Args:
          metric: a function accepting (y_true, y_pred) and returning the evaluation metric(s)
          batch_size:
          num_workers:
          eval_train: if True, also compute the evaluation metrics on the training set
          save: save the json file to the output directory
        """
        if len(kwargs) > 0:
            logger.warn(f"Extra kwargs were provided to trainer.evaluate(): {kwargs}")
        # Save the complete model -> HACK
        self.seq_model.save(os.path.join(self.output_dir, 'seq_model.pkl'))

        # contruct a list of dataset to evaluate
        if eval_train:
            eval_datasets = [('train', self.train_dataset)] + self.valid_dataset
        else:
            eval_datasets = self.valid_dataset

        # skip some datasets for evaluation
        try:
            if len(eval_skip) > 0:
                logger.info(f"Using eval_skip: {eval_skip}")
                eval_datasets = [(k, v) for k, v in eval_datasets if k not in eval_skip]
        except Exception:
            logger.warn(f"eval datasets don't contain tuples. Unable to skip them using {eval_skip}")

        metric_res = OrderedDict()
        for d in eval_datasets:
            if len(d) == 2:
                dataset_name, dataset = d
                eval_metric = None  # Ignore the provided metric
            elif len(d) == 3:
                # specialized evaluation metric was passed
                dataset_name, dataset, eval_metric = d
            else:
                raise ValueError("Valid dataset needs to be a list of tuples of 2 or 3 elements"
                                 "(name, dataset) or (name, dataset, metric)")
            logger.info(f"Evaluating dataset: {dataset_name}")
            metric_res[dataset_name] = self.seq_model.evaluate(dataset,
                                                               eval_metric=eval_metric,
                                                               num_workers=num_workers,
                                                               batch_size=batch_size)
        if save:
            write_json(metric_res, self.evaluation_path, indent=2)
            logger.info("Saved metrics to {}".format(self.evaluation_path))

        if self.cometml_experiment is not None:
            self.cometml_experiment.log_metrics(flatten(metric_res, separator='/'), prefix="eval/")

        if self.wandb_run is not None:
            self.wandb_run.summary.update(flatten(dict_prefix_key(metric_res, prefix="eval/"), separator='/'))

        return metric_res
