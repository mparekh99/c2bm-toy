from typing import Any, Optional, Mapping, Type
import pickle
import itertools

import torch
from torch import nn
from torchmetrics import Metric, MetricCollection
from torchmetrics.collections import _remove_prefix
import pytorch_lightning as pl

from src.models.layers.intervention import get_test_intervention_index

class Predictor(pl.LightningModule):    
    def __init__(self,
                model: Optional[nn.Module] = None,
                metrics: Optional[Mapping[str, Metric]] = None,
                optim_class: Optional[Type] = None,
                optim_kwargs: Optional[Mapping] = None,
                scheduler_class: Optional[Type] = None,
                scheduler_kwargs: Optional[Mapping] = None,
                intervention_prob: Optional[float] = 0.2,
                c_names: Optional[list] = None,
                test_interv_policy: Optional[str] = None,
                test_interv_noise: Optional[float] = 0.,
                ):
        super(Predictor, self).__init__()         
        self.model = model
        self.save_hyperparameters(ignore=["model"], logger=False)

        self.optim_class = optim_class
        self.optim_kwargs = optim_kwargs or dict()
        self.scheduler_class = scheduler_class
        self.scheduler_kwargs = scheduler_kwargs or dict()

        # for regularization
        self.intervention_prob = intervention_prob
        # store the intervention policy
        self.test_interv_policy = test_interv_policy
        self.test_interv_noise = test_interv_noise  

        self.c_names = c_names
        self.n_concepts = len(c_names)

        if metrics is None:
            metrics = dict()
        self._set_metrics(metrics)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    @staticmethod
    def _check_metric(metric):
        metric = metric.clone()
        metric.reset()
        return metric
    
    def _set_metrics(self, metrics):
        # --- accuracy metrics ---
        y_acc_metrics = {'y_accuracy': metrics.get('classification_acc')}
        c_acc_metrics = {k: metrics.get('classification_acc') for k in self.c_names}

        # task accuracy metrics
        self.train_y_metrics = MetricCollection(
            metrics={k: self._check_metric(m) for k, m in y_acc_metrics.items()},
            prefix="train/y/")
        self.val_y_metrics = MetricCollection(
            metrics={k: self._check_metric(m) for k, m in y_acc_metrics.items()},
            prefix="val/y/")
        self.test_y_metrics = MetricCollection(
            metrics={k: self._check_metric(m) for k, m in y_acc_metrics.items()},
            prefix="test/y/")
        
        # --- concept accuracy metrics ---
        self.train_c_metrics = MetricCollection(
            metrics={k: self._check_metric(m) for k, m in c_acc_metrics.items()},
            prefix="train/c/")
        self.val_c_metrics = MetricCollection(
            metrics={k: self._check_metric(m) for k, m in c_acc_metrics.items()},
            prefix="val/c/")
        self.test_c_metrics = MetricCollection(
            metrics={k: self._check_metric(m) for k, m in c_acc_metrics.items()},
            prefix="test/c/")      
          
        if self.model.has_concepts:
            # --- ground truth intervention metrics ---
            c_acc_metrics['_baseline'] = metrics.get('classification_acc')
            c_acc_levels_metrics = {f'level {n}': metrics.get('classification_acc')
                                    for n in range(0, len(self.test_interv_policy)+1)}
            
            # task accuracy after invervention on each individual concept 
            # (one metric for each concept)
            self.test_intervention_single_y = MetricCollection(
                metrics={k: self._check_metric(m) for k, m in c_acc_metrics.items()},
                prefix="test_intervention/single/y/")
            
            # task accuracy after intervention of each graph level
            self.test_intervention_level_y = MetricCollection(
                metrics={k: self._check_metric(m) for k, m in c_acc_levels_metrics.items()},
                prefix="test_intervention/level/y/")

            # # individual child concept accuracy after 
            # # intervention on ancestors in the graph
            # childs_per_level = {}
            # for l in range(0, len(self.test_interv_policy)+1):
            #     childs = list(itertools.chain(*self.test_interv_policy[l:]))
            #     for child in childs:
            #         child_name = self.c_names[child]
            #         childs_per_level[f'level {l}/child {child_name}'] = metrics.get('classification_acc')
            # self.test_intervention_level_c = MetricCollection(
            #     metrics={k: self._check_metric(m) for k, m in childs_per_level.items()},
            #     prefix="test_intervention/level/c/")

            # individual concept accuracy (task ancestors only, according to the policy) after 
            # intervention on levels defined by the policy
            nodes_per_level = {}
            indices_in_policy = list(itertools.chain(*self.test_interv_policy))
            c_names_in_policy = [self.c_names[i] for i in indices_in_policy]
            nodes_per_level.update({
                f'level {l}/node {c}': metrics.get('classification_acc')
                for l in range(len(self.test_interv_policy) + 1)
                for c in c_names_in_policy
            })
            self.test_intervention_level_c = MetricCollection(
                metrics={k: self._check_metric(m) for k, m in nodes_per_level.items()},
                prefix="test_intervention/level/c/")

            cace_task_concepts = [
                'Wind_Present',
                'Cup_Overfill',
                'Grip_Stability',
                'Spill_Risk',
                'Path_Clear',
                'Laptop_Nearby',
                'Table_Clutter',
                'Flat_Surface',
                'Safe_Placement',
            ]   

            self.cace_single = MetricCollection(
                metrics={
                    f'{c}_to_Delivery_Feasibility':
                        self._check_metric(metrics.get('cace'))
                    for c in cace_task_concepts
                },
                prefix='test_intervention/cace_single/'
            )

            cace_blocking_pairs = [
                ('Cup_Overfill', 'Spill_Risk'),
                ('Wind_Present', 'Spill_Risk'),
                ('Grip_Stability', 'Spill_Risk'),
                ('Laptop_Nearby', 'Safe_Placement'),
                ('Table_Clutter', 'Safe_Placement'),
                ('Flat_Surface', 'Safe_Placement'),
            ]

            cace_metrics = {}

            for source, bottleneck in cace_blocking_pairs:
                pair_name = f'{source}_to_{bottleneck}'

                cace_metrics[f'{pair_name}/before'] = \
                    self._check_metric(metrics.get('cace'))

                cace_metrics[f'{pair_name}/after'] = \
                    self._check_metric(metrics.get('cace'))

            self.cace = MetricCollection(
                metrics=cace_metrics,
                prefix='test_intervention/cace/'
            )

            # # interv test metrics
            # --- fairness metrics ---
            self.cace_neighbor = MetricCollection(
                metrics = {'Cup_Overfill_before': self._check_metric(metrics.get('cace')),
                           'Grip_Stability_before': self._check_metric(metrics.get('cace')),
                           'Cup_Overfill_after': self._check_metric(metrics.get('cace')),
                           'Grip_Stability_after': self._check_metric(metrics.get('cace'))},   
                prefix="test_intervention/cace_neighborhood/")

            self.cace_fairness = MetricCollection(
                metrics={
                    'before': self._check_metric(metrics.get('cace')),
                    'after': self._check_metric(metrics.get('cace'))
                },
                prefix="test_intervention/cace_test/"
            )




    def log_metrics(self, metrics, **kwargs):
        """"""
        self.log_dict(
            metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True, **kwargs
        )

    def log_loss(self, name, loss, **kwargs):
        """"""
        self.log(
            name + "_loss",
            loss.detach(),
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=False,
            **kwargs,
        )

    def _unpack_batch(self, batch):
        """
        Unpack a batch into data and preprocessing dictionaries.
        """
        return batch['x'], batch['c'], batch['y']
    
    def on_after_batch_transfer(self, batch, dataloader_idx):
        # add batch_size to batch
        if isinstance(batch, dict):
            batch['batch_size'] = batch['x'].shape[0]
        else:
            raise NotImplementedError("Only dict batches are supported")
        return batch

    def get_intervention_index(self, c_shape, step):
        """
        Get intervention index for training time intervention.
        Args:
            c_shape: shape of the concept tensor
            step: (str) 'train' or 'val'
        """
        # for regularization only
        if step=='train':
            intervention_index = torch.bernoulli(torch.ones(c_shape) * self.intervention_prob)
        else:
            intervention_index = torch.zeros(c_shape)
        return intervention_index.to("cuda" if torch.cuda.is_available() else "cpu")

    
    def test_intervention(self, batch):
        if self.model.has_concepts:
            x, c, y = self._unpack_batch(batch)
            # maybe add noise
            if self.test_interv_noise > 0:
                x = x + torch.randn_like(x) * self.test_interv_noise

            # baseline task accuracy
            # do not intervene
            intervention_index = get_test_intervention_index(c.shape, [])
            inputs = {'x':x, 'c':c, 'intervention_index':intervention_index}
            # forward pass with intervention at test time
            y_output, c_output = self.forward(**inputs)
            y_hat, c_hat = self.model.filter_output_for_metric(y_output, c_output)
            # update metric after intervention:
            # how well can we predict y?
            self.test_intervention_single_y['_baseline'].update(y_hat, y)            

            # interventions on individual concepts
            for i, c_name_i in enumerate(self.c_names):
                if c_name_i in self.model.virtual_roots: continue
                # intervene on concept c_name_i
                intervention_index = get_test_intervention_index(c.shape, i)
                inputs = {'x':x, 'c':c, 'intervention_index':intervention_index}
                # forward pass with intervention at test time
                y_output, c_output = self.forward(**inputs)
                y_hat, c_hat = self.model.filter_output_for_metric(y_output, c_output)
                # update metric after intervention:
                # after interveening on concept c_name_i, how well can we predict y?
                self.test_intervention_single_y[c_name_i].update(y_hat, y)

            # level intervention
            for l in range(0, len(self.test_interv_policy)+1):
                nodes = list(itertools.chain(*self.test_interv_policy[:l]))
                intervention_index = get_test_intervention_index(c.shape, nodes)
                inputs = {'x':x, 'c':c, 'intervention_index':intervention_index}
                # forward pass with intervention at test time
                y_output, c_output = self.forward(**inputs)
                y_hat, c_hat = self.model.filter_output_for_metric(y_output, c_output)
                # update metric after intervention:
                # after interveening on a level of the policy, how well can we predict y?
                self.test_intervention_level_y[f'level {l}'].update(y_hat, y)
                # update metric after intervention:
                # after interveening on a level of the policy, how well can we predict each child concept?
                indices_in_policy = list(itertools.chain(*self.test_interv_policy))
                for node_index in indices_in_policy:
                    c_name = self.c_names[node_index]
                    if c_name in c_hat:
                        self.test_intervention_level_c[f'level {l}/node {c_name}'].update(c_hat[c_name], c[:,node_index])
                    else:
                        # if the concept is not in the output, we cannot compute the metric for that concept
                        # this can happen if the model does not predict all concepts
                        pass

    def concept_to_concept_fairness(self, batch, c1, c2, interupt):
        if self.model.has_concepts:
            x, c, y = self._unpack_batch(batch)

            i = self.c_names.index(c1)
            j = self.c_names.index(c2)
            k = self.c_names.index(interupt)

            interv_index, interv_values = get_test_intervention_index(c.shape, i, values=1)
            y_output, c_output = self.forward(**{'x':x, 'c':interv_values, 'intervention_index':interv_index})
            _, c_hat_before_do_1 = self.model.filter_output_for_metric(y_output, c_output)
            interv_index, interv_values = get_test_intervention_index(c.shape, i, values=0)
            y_output, c_output = self.forward(**{'x':x, 'c':interv_values, 'intervention_index':interv_index})
            _, c_hat_before_do_0 = self.model.filter_output_for_metric(y_output, c_output)
            self.cace_fairness['before'].update(c_hat_before_do_1[self.c_names[k]], c_hat_before_do_0[self.c_names[k]])

            interv_index, interv_values = get_test_intervention_index(c.shape, [j,i], values=[1,1])
            y_output, c_output = self.forward(**{'x':x, 'c':interv_values, 'intervention_index':interv_index})
            _, c_hat_after_do_1 = self.model.filter_output_for_metric(y_output, c_output)
            interv_index, interv_values = get_test_intervention_index(c.shape, [j,i], values=[1,0])
            y_output, c_output = self.forward(**{'x':x, 'c':interv_values, 'intervention_index':interv_index})

            _, c_hat_after_do_0 = self.model.filter_output_for_metric(y_output, c_output)
            self.cace_fairness['after'].update(c_hat_after_do_1[self.c_names[k]], c_hat_after_do_0[self.c_names[k]])  # # Node to compare against, replaces task node in this case
            
            # print(y_output, c_output)
            self.log_metrics(self.cace_fairness, batch_size=batch['batch_size'])



    def test_intervention_fairness(self, batch, c1, c2):
        if self.model.has_concepts:
            x, c, y = self._unpack_batch(batch)

            i = self.c_names.index(c1)
            j = self.c_names.index(c2)

            interv_index, interv_values = get_test_intervention_index(c.shape, i, values=1)
            y_output, c_output = self.forward(**{'x':x, 'c':interv_values, 'intervention_index':interv_index})
            y_hat_before_do_1, _ = self.model.filter_output_for_metric(y_output, c_output)
            interv_index, interv_values = get_test_intervention_index(c.shape, i, values=0)
            y_output, c_output = self.forward(**{'x':x, 'c':interv_values, 'intervention_index':interv_index})
            y_hat_before_do_0, _ = self.model.filter_output_for_metric(y_output, c_output)
            self.cace_fairness['before'].update(y_hat_before_do_1, y_hat_before_do_0)

            interv_index, interv_values = get_test_intervention_index(c.shape, [j,i], values=[1,1])
            y_output, c_output = self.forward(**{'x':x, 'c':interv_values, 'intervention_index':interv_index})
            y_hat_after_do_1, _ = self.model.filter_output_for_metric(y_output, c_output)
            interv_index, interv_values = get_test_intervention_index(c.shape, [j,i], values=[1,0])
            y_output, c_output = self.forward(**{'x':x, 'c':interv_values, 'intervention_index':interv_index})

            y_hat_after_do_0, _ = self.model.filter_output_for_metric(y_output, c_output)
            self.cace_fairness['after'].update(y_hat_after_do_1, y_hat_after_do_0)
            
            # print(y_output, c_output)
            self.log_metrics(self.cace_fairness, batch_size=batch['batch_size'])


    def cace_test(self, batch, c_1, c_2):

        if self.model.has_concepts:
            x, c, y = self._unpack_batch(batch)

            i = self.c_names.index(c_1)

            interv_index, interv_values = get_test_intervention_index(c.shape, i, values=1)
            y_output, c_output = self.forward(**{'x':x, 'c':interv_values, 'intervention_index':interv_index})
            y_hat_before_do_1, c_hat_before_do_1 = self.model.filter_output_for_metric(y_output, c_output)
            interv_index, interv_values = get_test_intervention_index(c.shape, i, values=0)
            # assert y_output == y_hat_before_do_1
            y_output, c_output = self.forward(**{'x':x, 'c':interv_values, 'intervention_index':interv_index})
            y_hat_before_do_0, c_hat_before_do_0 = self.model.filter_output_for_metric(y_output, c_output)
            # If arguement == task node
            # Determining a node's presence and absence through a forward pass of c2bm and then 
            # Compute the average effect on task node to be 1. 
            if c_2 == 'Delivery_Feasibility':
                # self.cace_single['cace_score'].update(y_hat_before_do_1, y_hat_before_do_0)

                metric_name = f'{c_1}_to_{c_2}'

                self.cace_single[metric_name].update(
                    y_hat_before_do_1,
                    y_hat_before_do_0
                )
                self.log_metrics(self.cace_single, batch_size=batch['batch_size'])
                return

            # self.cace['before'].update(y_hat_before_do_1, y_hat_before_do_0)
            pair_name = f'{c_1}_to_{c_2}'
            
            self.cace[f'{pair_name}/before'].update(
                y_hat_before_do_1,
                y_hat_before_do_0
            )

            # k = self.c_names.index('Cup_Overfill')
            # p = self.c_names.index('Grip_Stability')            

            # self.cace_neighbor['Cup_Overfill_before'].update(c_hat_before_do_1[self.c_names[k]], c_hat_before_do_0[self.c_names[k]])
            # self.cace_neighbor['Grip_Stability_before'].update(c_hat_before_do_1[self.c_names[p]], c_hat_before_do_0[self.c_names[p]])

            j = self.c_names.index(c_2)

            interv_index, interv_values = get_test_intervention_index(c.shape, [j,i], values=[1,1])
            y_output, c_output = self.forward(**{'x':x, 'c':interv_values, 'intervention_index':interv_index})
            y_hat_after_do_1, c_hat_after_do_1 = self.model.filter_output_for_metric(y_output, c_output)
            interv_index, interv_values = get_test_intervention_index(c.shape, [j,i], values=[1,0])
            y_output, c_output = self.forward(**{'x':x, 'c':interv_values, 'intervention_index':interv_index})

            y_hat_after_do_0, c_hat_after_do_0 = self.model.filter_output_for_metric(y_output, c_output)
            # self.cace['after'].update(y_hat_after_do_1, y_hat_after_do_0)
            self.cace[f'{pair_name}/after'].update(
                y_hat_after_do_1,
                y_hat_after_do_0
            )

            # self.cace_neighbor['Cup_Overfill_before'].update(c_hat_after_do_1[self.c_names[k]], c_hat_after_do_0[self.c_names[k]])
            # self.cace_neighbor['Grip_Stability_before'].update(c_hat_after_do_1[self.c_names[p]], c_hat_after_do_0[self.c_names[p]])


            # self.log_metrics(self.cace_neighbor, batch_size=batch['batch_size'])
            self.log_metrics(self.cace, batch_size=batch['batch_size'])


    def update_and_log_metrics(self, step, y_hat, y, c_hat, c, batch):
        # update and log task metrics
        y_collection = getattr(self, f"{step}_y_metrics")
        y_collection.update(y_hat, y)
        self.log_metrics(y_collection, batch_size=batch['batch_size'])
        # update and log concept metrics
        c_collection = getattr(self, f"{step}_c_metrics")
        # log metrics for all predicted concepts 
        # (the collection contains all concepts, but some models predicts only a subset)
        if c_hat is not None:
            for k, v in c_hat.items():  
                c_collection[k].update(v, c[:,self.c_names.index(k)])
        self.log_metrics(c_collection, batch_size=batch['batch_size'])

    def shared_step(self, batch, step):
        x, c, y = self._unpack_batch(batch)
        # print("Inside Shared Step -------")
        # for i in range(5):
        #     print(f"Sample {i}: concepts = {c[i]}    y = {y[i]}")

        intervention_index = self.get_intervention_index(c.shape, step=step)
        inputs = {'x':x, 'c':c, 'intervention_index':intervention_index}
        # model forward
        y_output, c_output = self.forward(**inputs)
        # Compute loss
        y_hat_loss, c_hat_loss = self.model.filter_output_for_loss(y_output, c_output)
        loss = self.model.loss(y_hat_loss, y, c_hat_loss, c)
        return loss, y_output, c_output, y, c

    def training_step(self, batch, batch_idx):
        loss, y_output, c_output, y, c = self.shared_step(batch, step='train')
        if torch.isnan(loss).any():
            print(f'at epoc: {self.current_epoch}, batch: {batch_idx}')
            print('Loss has nan')
        # Update metrics and log
        y_hat, c_hat = self.model.filter_output_for_metric(y_output, c_output)
        self.update_and_log_metrics("train", y_hat, y, c_hat, c, batch)
        self.log_loss("train", loss, batch_size=batch['batch_size'])
        return loss
    
    def on_train_epoch_end(self):
        # Set the current epoch for SCBM and update the list of concept probs for computing the concept percentiles
        if type(self.model).__name__ == 'SCBM':
            self.model.training_epoch = self.current_epoch
            # self.model.concept_pred = torch.cat(self.model.concept_pred_tmp, dim=0) 
            # self.model.concept_pred_tmp = []        

    def validation_step(self, batch, batch_idx):
        val_loss, y_output, c_output, y, c = self.shared_step(batch, step='val')
        # Update metrics and log
        y_hat, c_hat = self.model.filter_output_for_metric(y_output, c_output)
        self.update_and_log_metrics("val", y_hat, y, c_hat, c, batch)
        self.log_loss("val", val_loss, batch_size=batch['batch_size'])
        return val_loss
    
    def test_step(self, batch, batch_idx):
        test_loss, y_output, c_output, y, c = self.shared_step(batch, step='test')
        # Update metrics and log
        y_hat, c_hat = self.model.filter_output_for_metric(y_output, c_output)
        self.update_and_log_metrics("test", y_hat, y, c_hat, c, batch)
        self.log_loss("test", test_loss, batch_size=batch['batch_size'])
        # test-time interventions
        # self.test_intervention_fairness(batch, 'VENTLUNG', 'CATECHOL')
        # self.concept_to_concept_fairness(batch, 'KINKEDTUBE','VENTALV' , 'CATECHOL')
        # self.test_intervention(batch)
        # # Single cace
        # self.cace_test(batch, 'Wind_Present', 'Delivery_Feasibility')
        # self.cace_test(batch, 'Cup_Overfill', 'Delivery_Feasibility')
        # self.cace_test(batch, 'Grip_Stability', 'Delivery_Feasibility')
        # self.cace_test(batch, 'Spill_Risk', 'Delivery_Feasibility')
        # self.cace_test(batch, 'Path_Clear', 'Delivery_Feasibility')
        # self.cace_test(batch, 'Laptop_Nearby', 'Delivery_Feasibility')
        # self.cace_test(batch, 'Table_Clutter', 'Delivery_Feasibility')
        # self.cace_test(batch, 'Flat_Surface', 'Delivery_Feasibility')
        # self.cace_test(batch, 'Safe_Placement', 'Delivery_Feasibility')
        # # Block a node's effect on task node
        # self.cace_test(batch, 'Cup_Overfill', 'Spill_Risk')
        # self.cace_test(batch, 'Wind_Present', 'Spill_Risk')
        # self.cace_test(batch, 'Grip_Stability', 'Spill_Risk')
        # self.cace_test(batch, 'Laptop_Nearby', 'Safe_Placement')
        # self.cace_test(batch, 'Table_Clutter', 'Safe_Placement')
        # self.cace_test(batch, 'Flat_Surface', 'Safe_Placement')

        return test_loss

    def on_test_epoch_end(self):
        # baseline task accuracy
        y_baseline = self.test_y_metrics['y_accuracy'].compute().item()
        print(f"Baseline task accuracy: {y_baseline}")
        pickle.dump({'_baseline':y_baseline}, open(f'results/y_accuracy.pkl', 'wb'))

        # baseline concept accuracy
        c_baseline = {}
        for k, metric in self.test_c_metrics.items():
            k = _remove_prefix(k, self.test_c_metrics.prefix)
            c_baseline[k] = metric.compute().item()
            print(f"Baseline concept accuracy for {k}: {c_baseline[k]}")
        pickle.dump(c_baseline, open(f'results/c_accuracy.pkl', 'wb'))

        if self.model.has_concepts:
            # task accuracy after invervention on each individual concept
            y_int = {}
            for k, metric in self.test_intervention_single_y.items():
                c_name = _remove_prefix(k, self.test_intervention_single_y.prefix)
                y_int[c_name] = metric.compute().item()
                print(f"Task accuracy after intervention on {c_name}: {y_int[c_name]}")
            pickle.dump(y_int, open(f'results/single_c_interventions_on_y.pkl', 'wb'))

            # task accuracy after intervention of each policy level
            y_int = {}
            for k, metric in self.test_intervention_level_y.items():
                level = _remove_prefix(k, self.test_intervention_level_y.prefix)
                y_int[level] = metric.compute().item()
                print(f"Task accuracy after intervention on {level}: {y_int[level]}")
            pickle.dump(y_int, open(f'results/level_interventions_on_y.pkl', 'wb'))

            # individual concept accuracy after intervention of each policy level
            c_int = {}
            for k, metric in self.test_intervention_level_c.items():
                level = _remove_prefix(k, self.test_intervention_level_c.prefix)
                c_int[level] = metric.compute().item()
                print(f"Concept accuracy after intervention on {level}: {c_int[level]}")
            pickle.dump(c_int, open(f'results/level_interventions_on_c.pkl', 'wb'))

            # save graph and concepts
            pickle.dump({'concepts':self.c_names,
                         'policy':self.test_interv_policy}, open("graph.pkl", 'wb'))
            
            pickle.dump({'policy':self.test_interv_policy}, open("policy.pkl", 'wb'))

    def configure_optimizers(self):
        """"""
        cfg = dict()
        optimizer = self.optim_class(self.parameters(), **self.optim_kwargs)
        cfg["optimizer"] = optimizer
        if self.scheduler_class is not None:
            metric = self.scheduler_kwargs.pop("monitor", None)
            scheduler = self.scheduler_class(optimizer, **self.scheduler_kwargs)
            cfg["lr_scheduler"] = scheduler
            if metric is not None:
                cfg["monitor"] = metric
        return cfg
 