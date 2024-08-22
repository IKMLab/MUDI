import logging
import os
from typing import Literal

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.base.base_trainer import BaseTrainer
from src.models.dgat_conv_pretraining import DGatConvPretraining
from src.models.modeling_generator import PersonalizedDialogueGenerator
from src.utils.constants import COHERENCE_RELATIONS, ModelTrainMode
from src.utils.data_utils import save_dist_model, save_model
from src.utils.metrics import MetricsAccumulator
from src.utils.metrics.classification import (
    compute_accuracy,
    compute_f1,
)
from src.utils.metrics.ranking import (
    compute_hits_at_k,
    compute_mrr,
    compute_roc_auc,
)
from src.utils.model_outputs import (
    CoherenceAwarePersonalizedDialogueGenerationSeq2SeqTrainerOutput,
    DGatForCoherenceAwareDialogueEncodingOutput,
    DGatForCoherenceAwareDialogueEncodingTrainerOutput,
    DGatForPreTrainingOutput,
    DGatForPreTrainingTrainerOutput,
    PersonalizedDialogueGeneratorOutput,
    PersonalizedDialogueGeneratorTrainerOutput,
)
from src.utils.utils import scheduled_sampling_exp


class DGatForCoherenceAwareDialogueEncodingTrainer(BaseTrainer):
    r"""The trainer for the `DGatForCoherenceAwareDialogueEncoding`

    Args:
        config (dict): The configuration of the model
        logger (logging.Logger): The logger
        model (CoherenceGraphModel): The model
        train_data (DataLoader): The training data
        valid_data (DataLoader): The validation data
    """

    def __init__(self, config: dict, logger: logging.Logger,
                 model: DGatConvPretraining, train_data: DataLoader,
                 valid_data: DataLoader):
        self.loss_weight = {
            'coh_rel_cls': config.coh_rel_cls_weight,
            'link_prediction': config.link_prediction_weight,
            'next_resp_type_direct': config.next_resp_type_direct_weight,
            'next_resp_type_seq': config.next_resp_type_seq_weight,
        }

        self.use_resp_type_prediction_in_training = False

        super().__init__(config, logger, model, train_data, valid_data)

    @property
    def model_name(self) -> str:
        r"""Get the model name

        Returns:
            str: The model name
        """

        model_name = 'CoherenceAwareDialogueGraph_'
        model_name += f'{self.batch_size}bs_'
        model_name += f'{self.epochs}epochs_'
        model_name += f'{self.lr}lr_'
        model_name += f'{self.weight_decay}wd_'
        model_name += f'{self.learner}_'
        model_name += f'NLW{self.loss_weight["node"]}_'
        model_name += f'CRW{self.loss_weight["coh_rel_cls"]}_'
        model_name += f'LPW{self.loss_weight["link_prediction"]}_'
        model_name += f'NRTDW{self.loss_weight["next_resp_type_direct"]}_'
        model_name += f'NRTSW{self.loss_weight["next_resp_type_seq"]}'

        return model_name

    def training_step(
        self, data_loader: DataLoader
    ) -> DGatForCoherenceAwareDialogueEncodingTrainerOutput:
        self.model.train()

        metrics_accumulator = MetricsAccumulator(
            total_steps=len(data_loader),
            output_class=DGatForCoherenceAwareDialogueEncodingTrainerOutput)
        for batch_idx, batch in tqdm(enumerate(data_loader),
                                     desc='Training',
                                     total=len(data_loader)):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            output: DGatForCoherenceAwareDialogueEncodingOutput = self.model(
                **batch,
                use_resp_type_prediction_in_training=self.
                use_resp_type_prediction_in_training)

            losses: DGatForCoherenceAwareDialogueEncodingTrainerOutput = self.model.compute_loss(
                output, weight=self.loss_weight)
            total_loss = losses.total_loss

            self.optimizer.zero_grad()
            if self.distributed:
                self.accelerator.backward(total_loss)
            else:
                total_loss.backward()
            self.optimizer.step()

            metrics_accumulator.update(
                total_loss=total_loss.item(),
                coh_rel_cls_loss=losses.coh_rel_cls_loss.item(),
                link_prediction_loss=losses.link_prediction_loss.item(),
                next_resp_type_direct_loss=losses.next_resp_type_direct_loss.
                item(),
                next_resp_type_seq_loss=losses.next_resp_type_seq_loss.item(),
                coh_rel_accuracy=compute_accuracy(output.coh_rel_preds,
                                                  output.coh_rel_labels),
                coh_rel_f1=compute_f1(output.coh_rel_preds,
                                      output.coh_rel_labels),
                coh_rel_top_at_k=compute_hits_at_k(output.coh_rel_preds,
                                                   output.coh_rel_labels,
                                                   k=5),
                coh_rel_mrr=compute_mrr(output.coh_rel_preds,
                                        output.coh_rel_labels),
                link_pred_roc_auc=compute_roc_auc(
                    output.link_prediction_logits.view(-1).sigmoid(),
                    output.link_prediction_labels),
                next_resp_type_direct_f1=compute_f1(
                    output.next_resp_type_direct_preds,
                    output.next_resp_type_direct_labels),
                next_resp_type_direct_top_at_k=compute_hits_at_k(
                    output.next_resp_type_direct_preds,
                    output.next_resp_type_direct_labels,
                    k=5),
                next_resp_type_seq_f1=compute_f1(
                    output.next_resp_type_seq_preds,
                    output.next_resp_type_seq_labels),
                next_resp_type_seq_top_at_k=compute_hits_at_k(
                    output.next_resp_type_seq_preds,
                    output.next_resp_type_seq_labels,
                    k=5),
            )

            self.logger.info(f'''Training@bs{batch_idx}
                [Coherence Relations] predicted probability: {output.coh_rel_preds.tolist()[-1]} |
                [Coherence Relations] predicted labels: {[
                label for prob, label in zip(output.coh_rel_preds[-1],
                                             COHERENCE_RELATIONS) if prob > 0.5
            ]}''')
            self.logger.info(f'''Training@bs{batch_idx}
                [Coherence Relations] ground-truth one-hots: {output.coh_rel_labels.tolist()[-1]} |
                [Coherence Relations] ground-truth labels: {[
                label for prob, label in zip(output.coh_rel_labels[-1],
                                             COHERENCE_RELATIONS) if prob == 1.0
            ]}''')
            self.logger.info(f'''Training@bs{batch_idx}
                [Next Response Types (Directly)] predicted probability: {output.next_resp_type_direct_preds.tolist()[-1]} |
                [Next Response Types (Directly)] predicted labels: {[
                label for prob, label in zip(output.next_resp_type_direct_preds[-1],
                                             COHERENCE_RELATIONS) if prob > 0.5
            ]}''')
            self.logger.info(f'''Training@bs{batch_idx}
                [Next Response Types (Directly)] ground-truth one-hots: {output.next_resp_type_direct_labels.tolist()[-1]} |
                [Next Response Types (Directly)] ground-truth labels: {[
                label for prob, label in zip(output.next_resp_type_direct_labels[-1],
                                             COHERENCE_RELATIONS) if prob == 1.0
            ]}''')
            self.logger.info(f'''Training@bs{batch_idx}
                [Next Response Types (Sequential)] predicted probability: {output.next_resp_type_seq_preds.tolist()[-1]} |
                [Next Response Types (Sequential)] predicted labels: {[
                label for prob, label in zip(output.next_resp_type_seq_preds[-1],
                                             COHERENCE_RELATIONS) if prob > 0.5
            ]}''')
            self.logger.info(f'''Training@bs{batch_idx}
                [Next Response Types (Sequential)] ground-truth one-hots: {output.next_resp_type_seq_labels.tolist()[-1]} |
                [Next Response Types (Sequential)] ground-truth labels: {[
                label for prob, label in zip(output.next_resp_type_seq_labels[-1],
                                             COHERENCE_RELATIONS) if prob == 1.0
            ]}''')

        return metrics_accumulator.averages()

    def evaluate_step(
        self, data_loader: DataLoader
    ) -> DGatForCoherenceAwareDialogueEncodingTrainerOutput:
        self.model.eval()

        metrics_accumulator = MetricsAccumulator(
            total_steps=len(data_loader),
            output_class=DGatForCoherenceAwareDialogueEncodingTrainerOutput)
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(data_loader),
                                         desc='Validation',
                                         total=len(data_loader)):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                output: DGatForCoherenceAwareDialogueEncodingOutput = self.model(
                    **batch)

                losses: DGatForCoherenceAwareDialogueEncodingTrainerOutput = self.model.compute_loss(
                    output, weight=self.loss_weight)

                metrics_accumulator.update(
                    total_loss=losses.total_loss.item(),
                    coh_rel_cls_loss=losses.coh_rel_cls_loss.item(),
                    link_prediction_loss=losses.link_prediction_loss.item(),
                    coh_rel_accuracy=compute_accuracy(output.coh_rel_preds,
                                                      output.coh_rel_labels),
                    next_resp_type_direct_loss=losses.
                    next_resp_type_direct_loss.item(),
                    next_resp_type_seq_loss=losses.next_resp_type_seq_loss.item(
                    ),
                    coh_rel_f1=compute_f1(output.coh_rel_preds,
                                          output.coh_rel_labels),
                    coh_rel_top_at_k=compute_hits_at_k(output.coh_rel_preds,
                                                       output.coh_rel_labels,
                                                       k=5),
                    coh_rel_mrr=compute_mrr(output.coh_rel_preds,
                                            output.coh_rel_labels),
                    link_pred_roc_auc=compute_roc_auc(
                        output.link_prediction_logits.view(-1).sigmoid(),
                        output.link_prediction_labels),
                    next_resp_type_direct_f1=compute_f1(
                        output.next_resp_type_direct_preds,
                        output.next_resp_type_direct_labels),
                    next_resp_type_direct_top_at_k=compute_hits_at_k(
                        output.next_resp_type_direct_preds,
                        output.next_resp_type_direct_labels,
                        k=5),
                    next_resp_type_seq_f1=compute_f1(
                        output.next_resp_type_seq_preds,
                        output.next_resp_type_seq_labels),
                    next_resp_type_seq_top_at_k=compute_hits_at_k(
                        output.next_resp_type_seq_preds,
                        output.next_resp_type_seq_labels,
                        k=5),
                )

            self.logger.info(f'''Validation@bs{batch_idx}
                [Coherence Relations] predicted probability: {output.coh_rel_preds.tolist()[-1]} |
                [Coherence Relations] predicted labels: {[
                label for prob, label in zip(output.coh_rel_preds[-1],
                                             COHERENCE_RELATIONS) if prob > 0.5
            ]}''')
            self.logger.info(f'''Validation@bs{batch_idx}
                [Coherence Relations] ground-truth one-hots: {output.coh_rel_labels.tolist()[-1]} |
                [Coherence Relations] ground-truth labels: {[
                label for prob, label in zip(output.coh_rel_labels[-1],
                                             COHERENCE_RELATIONS) if prob == 1.0
            ]}''')
            self.logger.info(f'''Validation@bs{batch_idx}
                [Next Response Types (Directly)] predicted probability: {output.next_resp_type_direct_preds.tolist()[-1]} |
                [Next Response Types (Directly)] predicted labels: {[
                label for prob, label in zip(output.next_resp_type_direct_preds[-1],
                                             COHERENCE_RELATIONS) if prob > 0.5
            ]}''')
            self.logger.info(f'''Validation@bs{batch_idx}
                [Next Response Types (Directly)] ground-truth one-hots: {output.next_resp_type_direct_labels.tolist()[-1]} |
                [Next Response Types (Directly)] ground-truth labels: {[
                label for prob, label in zip(output.next_resp_type_direct_labels[-1],
                                             COHERENCE_RELATIONS) if prob == 1.0
            ]}''')
            self.logger.info(f'''Validation@bs{batch_idx}
                [Next Response Types (Sequential)] predicted probability: {output.next_resp_type_seq_preds.tolist()[-1]} |
                [Next Response Types (Sequential)] predicted labels: {[
                label for prob, label in zip(output.next_resp_type_seq_preds[-1],
                                             COHERENCE_RELATIONS) if prob > 0.5
            ]}''')
            self.logger.info(f'''Validation@bs{batch_idx}
                [Next Response Types (Sequential)] ground-truth one-hots: {output.next_resp_type_seq_labels.tolist()[-1]} |
                [Next Response Types (Sequential)] ground-truth labels: {[
                label for prob, label in zip(output.next_resp_type_seq_labels[-1],
                                             COHERENCE_RELATIONS) if prob == 1.0
            ]}''')

        return metrics_accumulator.averages()

    def predict(self, data_loader: DataLoader) -> list[dict[str, list[str]]]:
        self.model.eval()

        coh_rel_pred_results = []
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(data_loader),
                                         desc='Test',
                                         total=len(data_loader)):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                output: DGatForCoherenceAwareDialogueEncodingOutput = self.model(
                    **batch)

                for probs in output.coh_rel_preds:
                    pred_labels = [
                        label for label, prob in zip(COHERENCE_RELATIONS, probs)
                        if prob > 0.5
                    ]
                    coh_rel_pred_results.append({
                        'pred_labels': pred_labels,
                        'pred_probs': probs.tolist()
                    })

        return coh_rel_pred_results

    def train(self) -> tuple[str, int]:
        model_path = ''
        endure_count = 0
        best_epoch = 0
        best_val_loss = float('inf')
        for epoch in tqdm(range(1, self.epochs + 1)):
            self.logger.info(f'epoch {epoch}')

            self.use_resp_type_prediction_in_training = scheduled_sampling_exp(
                epoch - 1, self.epochs)

            # Training
            train_output = self.training_step(self.train_data)
            self.logger.info(
                f'''[Coherence Relations Classification] loss: {train_output.coh_rel_cls_loss:4.4f} |
                [Link Prediction] loss: {train_output.link_prediction_loss:4.4f} |
                [Next Response Types (Directly)] loss: {train_output.next_resp_type_direct_loss:4.4f} |
                [Next Response Types (Sequential)] loss: {train_output.next_resp_type_seq_loss:4.4f} on Training'''
            )
            self.logger.info(
                f'''[Coherence Relations Classification] accuracy: {train_output.coh_rel_accuracy:4.4f} |
                [Coherence Relations Classification] f1: {train_output.coh_rel_f1:4.4f} |
                [Coherence Relations Classification] top@5: {train_output.coh_rel_top_at_k:4.4f} |
                [Coherence Relations Classification] mrr: {train_output.coh_rel_mrr:4.4f} |
                [Link Prediction] roc-auc: {train_output.link_pred_roc_auc:4.4f} |
                [Next Response Types (Directly)] f1: {train_output.next_resp_type_direct_f1:4.4f} |
                [Next Response Types (Directly)] top@5: {train_output.next_resp_type_direct_top_at_k:4.4f} |
                [Next Response Types (Sequential)] f1: {train_output.next_resp_type_seq_f1:4.4f} |
                [Next Response Types (Sequential)] top@5: {train_output.next_resp_type_seq_top_at_k:4.4f} on Training'''
            )

            # Validation
            valid_output = self.evaluate_step(self.valid_data)
            self.logger.info(
                f'''[Coherence Relations Classification] loss: {valid_output.coh_rel_cls_loss:4.4f} |
                [Link Prediction] loss: {valid_output.link_prediction_loss:4.4f} |
                [Next Response Types (Directly)] loss: {valid_output.next_resp_type_direct_loss:4.4f} |
                [Next Response Types (Sequential)] loss: {valid_output.next_resp_type_seq_loss:4.4f} on Validation'''
            )
            self.logger.info(
                f'''[Coherence Relations Classification] accuracy: {valid_output.coh_rel_accuracy:4.4f} |
                [Coherence Relations Classification] f1: {valid_output.coh_rel_f1:4.4f} |
                [Coherence Relations Classification] top@5: {valid_output.coh_rel_top_at_k:4.4f} |
                [Coherence Relations Classification] mrr: {valid_output.coh_rel_mrr:4.4f} |
                [Link Prediction] roc-auc: {valid_output.link_pred_roc_auc:4.4f} |
                [Next Response Types (Directly)] f1: {valid_output.next_resp_type_direct_f1:4.4f} |
                [Next Response Types (Directly)] top@5: {valid_output.next_resp_type_direct_top_at_k:4.4f} |
                [Next Response Types (Sequential)] f1: {valid_output.next_resp_type_seq_f1:4.4f} |
                [Next Response Types (Sequential)] top@5: {valid_output.next_resp_type_seq_top_at_k:4.4f} on Validation'''
            )

            if valid_output.total_loss < best_val_loss:
                endure_count = 0

                saved_file_prefix = f'{self.model_name}-{self.dataset_name}'
                saved_model_file = f'{saved_file_prefix}_model.pt'
                saved_state_dict = f'{saved_file_prefix}_state_dict.pt'
                model_path = os.path.join(self.checkpoint, saved_model_file)
                state_dict_path = os.path.join(self.checkpoint,
                                               saved_state_dict)

                if not self.distributed:
                    save_model(self.model, model_path)
                    self.logger.info(
                        '(resume training ok) Save the best model' + model_path)
                    save_model(self.model.state_dict(), state_dict_path)
                    self.logger.info(
                        "(inference ok) Save the best model's state dict" +
                        state_dict_path)
                else:
                    save_dist_model(self.accelerator, self.model,
                                    state_dict_path)
                    self.logger.info(
                        "(inference ok) Save the best model's state dict" +
                        state_dict_path)

                best_val_loss = valid_output.total_loss
                best_epoch = epoch
            else:
                endure_count += 1
                self.logger.info(f'Endured {endure_count} time(s)')
                if endure_count == self.endure_times:
                    self.logger.info(
                        'Cannot endure it anymore | Exiting from early stop')
                    break

                self.scheduler.step()
                self.logger.info(
                    f'Learning rate set to {self.scheduler.get_last_lr()[0]:2.8f}'
                )

        return model_path, best_epoch


class DGatForPreTrainingTrainer(BaseTrainer):
    r"""The trainer for pretraining the `DGatForPreTraining`

    Args:
        config (dict): The configuration of the model
        logger (logging.Logger): The logger
        model (CoherenceGraphModel): The model
        train_data (DataLoader): The training data
        valid_data (DataLoader): The validation data
    """

    def __init__(self, config: dict, logger: logging.Logger,
                 model: DGatConvPretraining, train_data: DataLoader,
                 valid_data: DataLoader):
        super().__init__(config, logger, model, train_data, valid_data)

    @property
    def model_name(self) -> str:
        r"""Get the model name

        Returns:
            str: The model name
        """

        model_name = 'PretrainedDialogueGraph_'
        model_name += f'{self.batch_size}bs_'
        model_name += f'{self.epochs}epochs_'
        model_name += f'{self.lr}lr_'
        model_name += f'{self.weight_decay}wd_'
        model_name += f'{self.learner}'

        return model_name

    def training_step(
            self, data_loader: DataLoader) -> DGatForPreTrainingTrainerOutput:
        self.model.train()

        metrics_accumulator = MetricsAccumulator(
            total_steps=len(data_loader),
            output_class=DGatForPreTrainingTrainerOutput)
        for batch_idx, batch in tqdm(enumerate(data_loader),
                                     desc='Training',
                                     total=len(data_loader)):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            output: DGatForPreTrainingOutput = self.model(**batch)

            losses = self.model.compute_loss(output)
            total_loss = losses.total_loss

            self.optimizer.zero_grad()
            if self.distributed:
                self.accelerator.backward(total_loss)
            else:
                total_loss.backward()
            self.optimizer.step()

            metrics_accumulator.update(
                total_loss=total_loss.item(),
                path_loss=losses.path_loss.item(),
                turn_loss=losses.turn_loss.item(),
                adj_recon_loss=losses.adj_recon_loss.item())

        return metrics_accumulator.averages()

    def evaluate_step(
            self, data_loader: DataLoader) -> DGatForPreTrainingTrainerOutput:
        self.model.eval()

        metrics_accumulator = MetricsAccumulator(
            total_steps=len(data_loader),
            output_class=DGatForPreTrainingTrainerOutput)
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(data_loader),
                                         desc='Validation',
                                         total=len(data_loader)):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                output: DGatForPreTrainingOutput = self.model(**batch)

                losses = self.model.compute_loss(output)

                metrics_accumulator.update(
                    total_loss=losses.total_loss.item(),
                    path_loss=losses.path_loss.item(),
                    turn_loss=losses.turn_loss.item(),
                    adj_recon_loss=losses.adj_recon_loss.item())

        return metrics_accumulator.averages()

    def train(self) -> tuple[str, int]:
        model_path = ''
        endure_count = 0
        best_epoch = 0
        best_val_loss = float('inf')
        for epoch in tqdm(range(1, self.epochs + 1)):
            self.logger.info(f'epoch {epoch}')

            # Training
            train_output = self.training_step(self.train_data)
            self.logger.info(f'''[Path] loss: {train_output.path_loss:4.4f} |
                [Turn] loss: {train_output.turn_loss:4.4f} |
                [Adjacency Matrix Reconstruction] loss: {train_output.adj_recon_loss:4.4f} on Training'''
                             )

            # Validation
            valid_output = self.evaluate_step(self.valid_data)
            self.logger.info(f'''[Path] loss: {valid_output.path_loss:4.4f} |
                [Turn] loss: {valid_output.turn_loss:4.4f} |
                [Adjacency Matrix Reconstruction] loss: {valid_output.adj_recon_loss:4.4f} on Validation'''
                             )

            if valid_output.total_loss < best_val_loss:
                endure_count = 0

                saved_file_prefix = f'{self.model_name}-{self.dataset_name}'
                saved_model_file = f'{saved_file_prefix}_model.pt'
                saved_state_dict = f'{saved_file_prefix}_state_dict.pt'
                model_path = os.path.join(self.checkpoint, saved_model_file)
                state_dict_path = os.path.join(self.checkpoint,
                                               saved_state_dict)

                if not self.distributed:
                    save_model(self.model, model_path)
                    self.logger.info(
                        '(resume training ok) Save the best model' + model_path)
                    save_model(self.model.state_dict(), state_dict_path)
                    self.logger.info(
                        "(inference ok) Save the best model's state dict" +
                        state_dict_path)
                else:
                    save_dist_model(self.accelerator, self.model,
                                    state_dict_path)
                    self.logger.info(
                        "(inference ok) Save the best model's state dict" +
                        state_dict_path)

                best_val_loss = valid_output.total_loss
                best_epoch = epoch
            else:
                endure_count += 1
                self.logger.info(f'Endured {endure_count} time(s)')
                if endure_count == self.endure_times:
                    self.logger.info(
                        'Cannot endure it anymore | Exiting from early stop')
                    break

                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                self.scheduler.step()
                self.logger.info(
                    f'Learning rate set to {self.scheduler.get_last_lr()[0]:2.8f}'
                )

        return model_path, best_epoch


class PersonalizedDialogueGeneratorTrainer(BaseTrainer):
    r"""The trainer for the `PersonalizedDialogueGenerator`

    Args:
        config (dict): The configuration of the model
        logger (logging.Logger): The logger
        model (CoherenceGraphModel): The model
        train_data (DataLoader): The training data
        valid_data (DataLoader): The validation data
    """

    def __init__(self, config: dict, logger: logging.Logger,
                 model: PersonalizedDialogueGenerator, train_data: DataLoader,
                 valid_data: DataLoader):
        super().__init__(config, logger, model, train_data, valid_data)

        self.top_k = 3
        self.tau = 0.2
        self.use_resp_type_prediction_in_training = False

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=2,
            cooldown=2,
            factor=0.9,
            threshold=1e-4,
            min_lr=1e-7)

    @property
    def model_name(self) -> str:
        r"""Get the model name

        Returns:
            str: The model name
        """

        model_name = 'PersonalizedDialogueGenerator_'
        model_name += f'{self.batch_size}bs_'
        model_name += f'{self.epochs}epochs_'
        model_name += f'{self.lr}lr_'
        model_name += f'{self.weight_decay}wd_'
        model_name += f'{self.learner}_'
        model_name += f'top-{self.top_k}'

        return model_name

    def training_step(
            self, data_loader: DataLoader
    ) -> PersonalizedDialogueGeneratorTrainerOutput:
        self.model.train()

        metrics_accumulator = MetricsAccumulator(
            total_steps=len(data_loader),
            output_class=PersonalizedDialogueGeneratorTrainerOutput)
        for batch_idx, batch in tqdm(enumerate(data_loader),
                                     desc='Training',
                                     total=len(data_loader)):

            batch['dialogue_encoder_input'] = {
                k: v.to(self.device)
                for k, v in batch['dialogue_encoder_input'].items()
            }
            batch['generator_input'] = {
                k: v.to(self.device)
                for k, v in batch['generator_input'].items()
            }
            output: PersonalizedDialogueGeneratorOutput = self.model(
                dialogue_encoder_input=batch['dialogue_encoder_input'],
                generator_input=batch['generator_input'],
                top_k=self.top_k,
                tau=self.tau,
                use_resp_type_prediction_in_training= \
                    self.use_resp_type_prediction_in_training,
                is_generation=False)
            losses = self.model.compute_loss(output)
            total_loss = losses.total_loss

            self.optimizer.zero_grad()
            if self.distributed:
                self.accelerator.backward(total_loss)
            else:
                total_loss.backward()
            self.optimizer.step()

            # multiple_choice_preds = torch.sigmoid(output.cls_logits)
            multiple_choice_preds = torch.argmax(torch.softmax(
                output.cls_logits, dim=1),
                                                 dim=1)
            multiple_choice_labels = output.cls_labels

            metrics_accumulator.update(
                total_loss=total_loss.item(),
                nll_loss=losses.nll_loss.item(),
                cls_loss=losses.cls_loss.item(),
                multiple_choice_accuracy=compute_accuracy(
                    multiple_choice_preds,
                    multiple_choice_labels,
                    cls_type='multiclass'),
                multiple_choice_f1=compute_f1(multiple_choice_preds,
                                              multiple_choice_labels,
                                              cls_type='multiclass'),
            )

        return metrics_accumulator.averages()

    def evaluate_step(
            self, data_loader: DataLoader
    ) -> PersonalizedDialogueGeneratorTrainerOutput:
        self.model.eval()

        metrics_accumulator = MetricsAccumulator(
            total_steps=len(data_loader),
            output_class=PersonalizedDialogueGeneratorTrainerOutput)
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(data_loader),
                                         desc='Validation',
                                         total=len(data_loader)):
                batch['dialogue_encoder_input'] = {
                    k: v.to(self.device)
                    for k, v in batch['dialogue_encoder_input'].items()
                }
                batch['generator_input'] = {
                    k: v.to(self.device)
                    for k, v in batch['generator_input'].items()
                }
                output: PersonalizedDialogueGeneratorOutput = self.model(
                    dialogue_encoder_input=batch['dialogue_encoder_input'],
                    generator_input=batch['generator_input'],
                    top_k=self.top_k,
                    tau=self.tau,
                    is_generation=False)
                losses = self.model.compute_loss(output)

                # multiple_choice_preds = torch.sigmoid(output.cls_logits)
                multiple_choice_preds = torch.argmax(torch.softmax(
                    output.cls_logits, dim=1),
                                                     dim=1)
                multiple_choice_labels = output.cls_labels

                metrics_accumulator.update(
                    total_loss=losses.total_loss.item(),
                    nll_loss=losses.nll_loss.item(),
                    cls_loss=losses.cls_loss.item(),
                    multiple_choice_accuracy=compute_accuracy(
                        multiple_choice_preds,
                        multiple_choice_labels,
                        cls_type='multiclass'),
                    multiple_choice_f1=compute_f1(multiple_choice_preds,
                                                  multiple_choice_labels,
                                                  cls_type='multiclass'),
                )

        return metrics_accumulator.averages()

    def train(self) -> tuple[str, int]:
        model_path = ''
        endure_count = 0
        best_epoch = 0
        best_val_loss = float('inf')
        for epoch in tqdm(range(1, self.epochs + 1)):
            self.logger.info(f'epoch {epoch}')

            self.use_resp_type_prediction_in_training = scheduled_sampling_exp(
                epoch - 1, self.epochs)

            # Training
            train_output = self.training_step(self.train_data)
            # self.logger.info(f'''[NLL] loss: {train_output.nll_loss:4.4f} on Training'''
            #                  )
            self.logger.info(f'''[NLL] loss: {train_output.nll_loss:4.4f} |
                [Multiple Choice Classification] loss:  {train_output.cls_loss:4.4f} on Training'''
                             )
            self.logger.info(
                f'''[Multiple Choice Classification] accuracy: {train_output.multiple_choice_accuracy:4.4f} |
                [Multiple Choice Classification] f1: {train_output.multiple_choice_f1:4.4f} on Training'''
            )

            # Validation
            valid_output = self.evaluate_step(self.valid_data)
            # self.logger.info(f'''[NLL] loss: {valid_output.nll_loss:4.4f} on Validation'''
            #                  )
            self.logger.info(f'''[NLL] loss: {valid_output.nll_loss:4.4f} |
                [Multiple Choice Classification] loss:  {valid_output.cls_loss:4.4f} on Validation'''
                             )
            self.logger.info(
                f'''[Multiple Choice Classification] accuracy: {valid_output.multiple_choice_accuracy:4.4f} |
                [Multiple Choice Classification] f1: {valid_output.multiple_choice_f1:4.4f} on Validation'''
            )

            self.scheduler.step(valid_output.total_loss)
            self.logger.info(
                f'Learning rate set to {self.scheduler.get_last_lr()[0]:2.8f}')

            if valid_output.total_loss < best_val_loss:
                endure_count = 0

                saved_file_prefix = f'{self.model_name}-{self.dataset_name}'
                saved_model_file = f'{saved_file_prefix}_model.pt'
                saved_state_dict = f'{saved_file_prefix}_state_dict.pt'
                model_path = os.path.join(self.checkpoint, saved_model_file)
                state_dict_path = os.path.join(self.checkpoint,
                                               saved_state_dict)

                if not self.distributed:
                    self.model.save_pretrained(self.checkpoint / 'model',
                                               safe_serialization=False)
                    save_model(self.model, model_path)
                    self.logger.info(
                        '(resume training ok) Save the best model' + model_path)
                    save_model(self.model.state_dict(), state_dict_path)
                    self.logger.info(
                        "(inference ok) Save the best model's state dict" +
                        state_dict_path)
                else:
                    self.model.save_pretrained(self.checkpoint / 'model',
                                               safe_serialization=False)
                    save_dist_model(self.accelerator, self.model,
                                    state_dict_path)
                    self.logger.info(
                        "(inference ok) Save the best model's state dict" +
                        state_dict_path)

                best_val_loss = valid_output.total_loss
                best_epoch = epoch
            else:
                endure_count += 1
                self.logger.info(f'Endured {endure_count} time(s)')
                if endure_count == self.endure_times:
                    self.logger.info(
                        'Cannot endure it anymore | Exiting from early stop')
                    break

        return model_path, best_epoch


class CoherenceAwarePersonalizedDialogueGenerationSeq2SeqTrainer(BaseTrainer):
    r"""The trainer for pretraining the `CoherenceAwarePersonalizedDialogueGenerationSeq2SeqTrainer`

    Args:
        config (dict): The configuration of the model
        logger (logging.Logger): The logger
        model (CoherenceGraphModel): The model
        train_data (DataLoader): The training data
        valid_data (DataLoader): The validation data
    """

    def __init__(self, config: dict, logger: logging.Logger,
                 model: PersonalizedDialogueGenerator, train_data: DataLoader,
                 valid_data: DataLoader):
        super().__init__(config, logger, model, train_data, valid_data)

        self.loss_weight = {
            'coh_rel_cls': config.coh_rel_cls_weight,
            'link_prediction': config.link_prediction_weight,
            'next_resp_type_direct': config.next_resp_type_direct_weight,
            'next_resp_type_seq': config.next_resp_type_seq_weight,
        }

        self.top_k = 3
        self.tau = 0.2
        self.use_resp_type_prediction_in_training = None

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=3,
            cooldown=2,
            factor=0.9,
            min_lr=1e-7)

    @property
    def model_name(self) -> str:
        r"""Get the model name

        Returns:
            str: The model name
        """

        model_name = 'PersonalizedDialogueGenerator_'
        model_name += f'{self.batch_size}bs_'
        model_name += f'{self.epochs}epochs_'
        model_name += f'{self.lr}lr_'
        model_name += f'{self.weight_decay}wd_'
        model_name += f'{self.learner}_'
        model_name += f'top-{self.top_k}'

        return model_name

    def training_step(
        self, data_loader: DataLoader
    ) -> CoherenceAwarePersonalizedDialogueGenerationSeq2SeqTrainerOutput:
        self.model.train()

        metrics_accumulator = MetricsAccumulator(
            total_steps=len(data_loader),
            output_class=
            CoherenceAwarePersonalizedDialogueGenerationSeq2SeqTrainerOutput)
        for batch_idx, batch in tqdm(enumerate(data_loader),
                                     desc='Training',
                                     total=len(data_loader)):

            batch['dialogue_encoder_input'] = {
                k: v.to(self.device)
                for k, v in batch['dialogue_encoder_input'].items()
            }
            batch['generator_input'] = {
                k: v.to(self.device)
                for k, v in batch['generator_input'].items()
            }
            output: PersonalizedDialogueGeneratorOutput = self.model(
                dialogue_encoder_input=batch['dialogue_encoder_input'],
                generator_input=batch['generator_input'],
                top_k=self.top_k,
                tau=self.tau,
                use_resp_type_prediction_in_training= \
                    self.use_resp_type_prediction_in_training,
                is_generation=False)
            losses = self.model.compute_loss(output, weight=self.loss_weight)
            total_loss = losses.total_loss

            self.optimizer.zero_grad()
            if self.distributed:
                self.accelerator.backward(total_loss)
            else:
                total_loss.backward()
            self.optimizer.step()

            coh_rel_preds = torch.sigmoid(output.cls_logits)
            coh_rel_labels = output.cls_labels

            metrics_accumulator.update(
                total_loss=total_loss.item(),
                nll_loss=losses.nll_loss.item(),
                coh_rel_cls_loss=losses.coh_rel_cls_loss.item(),
                coh_rel_accuracy=compute_accuracy(coh_rel_preds,
                                                  coh_rel_labels),
                coh_rel_f1=compute_f1(coh_rel_preds, coh_rel_labels),
                coh_rel_top_at_k=compute_hits_at_k(coh_rel_preds,
                                                   coh_rel_labels,
                                                   k=2),
                dial_encoder_coh_rel_cls_loss=losses.
                dial_encoder_coh_rel_cls_loss.item(),
                dial_encoder_link_prediction_loss=losses.
                dial_encoder_link_prediction_loss.item(),
                dial_encoder_next_resp_type_direct_loss=losses.
                dial_encoder_next_resp_type_direct_loss.item(),
                dial_encoder_next_resp_type_seq_loss=losses.
                dial_encoder_next_resp_type_seq_loss.item(),
                dial_encoder_coh_rel_accuracy=compute_accuracy(
                    output.dial_encoder_outputs.coh_rel_preds,
                    output.dial_encoder_outputs.coh_rel_labels),
                dial_encoder_coh_rel_f1=compute_f1(
                    output.dial_encoder_outputs.coh_rel_preds,
                    output.dial_encoder_outputs.coh_rel_labels),
                dial_encoder_coh_rel_top_at_k=compute_hits_at_k(
                    output.dial_encoder_outputs.coh_rel_preds,
                    output.dial_encoder_outputs.coh_rel_labels,
                    k=5),
                dial_encoder_link_pred_roc_auc=compute_roc_auc(
                    output.dial_encoder_outputs.link_prediction_logits.view(
                        -1).sigmoid(),
                    output.dial_encoder_outputs.link_prediction_labels),
                dial_encoder_next_resp_type_direct_f1=compute_f1(
                    output.dial_encoder_outputs.next_resp_type_direct_preds,
                    output.dial_encoder_outputs.next_resp_type_direct_labels),
                dial_encoder_next_resp_type_direct_top_at_k=compute_hits_at_k(
                    output.dial_encoder_outputs.next_resp_type_direct_preds,
                    output.dial_encoder_outputs.next_resp_type_direct_labels,
                    k=5),
                dial_encoder_next_resp_type_seq_f1=compute_f1(
                    output.dial_encoder_outputs.next_resp_type_seq_preds,
                    output.dial_encoder_outputs.next_resp_type_seq_labels),
                dial_encoder_next_resp_type_seq_top_at_k=compute_hits_at_k(
                    output.dial_encoder_outputs.next_resp_type_seq_preds,
                    output.dial_encoder_outputs.next_resp_type_seq_labels,
                    k=5),
            )

            self.logger.info(f'''Training@bs{batch_idx}
                [Coherence Relations] predicted probability: {coh_rel_preds.tolist()[-1]} |
                [Coherence Relations] predicted labels: {[
                label for prob, label in zip(coh_rel_preds[-1],
                                             COHERENCE_RELATIONS) if prob > 0.5
            ]}''')
            self.logger.info(f'''Training@bs{batch_idx}
                [Coherence Relations] ground-truth one-hots: {coh_rel_labels.tolist()[-1]} |
                [Coherence Relations] ground-truth labels: {[
                label for prob, label in zip(coh_rel_labels[-1],
                                             COHERENCE_RELATIONS) if prob == 1.0
            ]}''')
            self.logger.info(f'''Training@bs{batch_idx}
                [Dialogue Encoder Coherence Relations] predicted probability: {output.dial_encoder_outputs.coh_rel_preds.tolist()[-1]} |
                [Dialogue Encoder Coherence Relations] predicted labels: {[
                label for prob, label in zip(output.dial_encoder_outputs.coh_rel_preds[-1],
                                             COHERENCE_RELATIONS) if prob > 0.5
            ]}''')
            self.logger.info(f'''Training@bs{batch_idx}
                [Dialogue Encoder Coherence Relations] ground-truth one-hots: {output.dial_encoder_outputs.coh_rel_labels.tolist()[-1]} |
                [Dialogue Encoder Coherence Relations] ground-truth labels: {[
                label for prob, label in zip(output.dial_encoder_outputs.coh_rel_labels[-1],
                                             COHERENCE_RELATIONS) if prob == 1.0
            ]}''')
            self.logger.info(f'''Training@bs{batch_idx}
                [Dialogue Encoder Next Response Types (Directly)] predicted probability: {output.dial_encoder_outputs.next_resp_type_direct_preds.tolist()[-1]} |
                [Dialogue Encoder Next Response Types (Directly)] predicted labels: {[
                label for prob, label in zip(output.dial_encoder_outputs.next_resp_type_direct_preds[-1],
                                             COHERENCE_RELATIONS) if prob > 0.5
            ]}''')
            self.logger.info(f'''Training@bs{batch_idx}
                [Dialogue Encoder Next Response Types (Directly)] ground-truth one-hots: {output.dial_encoder_outputs.next_resp_type_direct_labels.tolist()[-1]} |
                [Dialogue Encoder Next Response Types (Directly)] ground-truth labels: {[
                label for prob, label in zip(output.dial_encoder_outputs.next_resp_type_direct_labels[-1],
                                             COHERENCE_RELATIONS) if prob == 1.0
            ]}''')
            self.logger.info(f'''Training@bs{batch_idx}
                [Dialogue Encoder Next Response Types (Sequential)] predicted probability: {output.dial_encoder_outputs.next_resp_type_seq_preds.tolist()[-1]} |
                [Dialogue Encoder Next Response Types (Sequential)] predicted labels: {[
                label for prob, label in zip(output.dial_encoder_outputs.next_resp_type_seq_preds[-1],
                                             COHERENCE_RELATIONS) if prob > 0.5
            ]}''')
            self.logger.info(f'''Training@bs{batch_idx}
                [Dialogue Encoder Next Response Types (Sequential)] ground-truth one-hots: {output.dial_encoder_outputs.next_resp_type_seq_labels.tolist()[-1]} |
                [Dialogue Encoder Next Response Types (Sequential)] ground-truth labels: {[
                label for prob, label in zip(output.dial_encoder_outputs.next_resp_type_seq_labels[-1],
                                             COHERENCE_RELATIONS) if prob == 1.0
            ]}''')

        return metrics_accumulator.averages()

    def evaluate_step(
        self, data_loader: DataLoader
    ) -> CoherenceAwarePersonalizedDialogueGenerationSeq2SeqTrainerOutput:
        self.model.eval()

        metrics_accumulator = MetricsAccumulator(
            total_steps=len(data_loader),
            output_class=
            CoherenceAwarePersonalizedDialogueGenerationSeq2SeqTrainerOutput)
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(data_loader),
                                         desc='Validation',
                                         total=len(data_loader)):
                batch['dialogue_encoder_input'] = {
                    k: v.to(self.device)
                    for k, v in batch['dialogue_encoder_input'].items()
                }
                batch['generator_input'] = {
                    k: v.to(self.device)
                    for k, v in batch['generator_input'].items()
                }
                output: DGatForPreTrainingOutput = self.model(
                    dialogue_encoder_input=batch['dialogue_encoder_input'],
                    generator_input=batch['generator_input'],
                    top_k=self.top_k,
                    tau=self.tau,
                    is_generation=False)
                losses = self.model.compute_loss(output,
                                                 weight=self.loss_weight)

                coh_rel_preds = torch.sigmoid(output.cls_logits)
                coh_rel_labels = output.cls_labels

                metrics_accumulator.update(
                    total_loss=losses.total_loss.item(),
                    nll_loss=losses.nll_loss.item(),
                    coh_rel_cls_loss=losses.coh_rel_cls_loss.item(),
                    coh_rel_accuracy=compute_accuracy(coh_rel_preds,
                                                      coh_rel_labels),
                    coh_rel_f1=compute_f1(coh_rel_preds, coh_rel_labels),
                    coh_rel_top_at_k=compute_hits_at_k(coh_rel_preds,
                                                       coh_rel_labels,
                                                       k=2),
                    dial_encoder_coh_rel_cls_loss=losses.
                    dial_encoder_coh_rel_cls_loss.item(),
                    dial_encoder_link_prediction_loss=losses.
                    dial_encoder_link_prediction_loss.item(),
                    dial_encoder_next_resp_type_direct_loss=losses.
                    dial_encoder_next_resp_type_direct_loss.item(),
                    dial_encoder_next_resp_type_seq_loss=losses.
                    dial_encoder_next_resp_type_seq_loss.item(),
                    dial_encoder_coh_rel_accuracy=compute_accuracy(
                        output.dial_encoder_outputs.coh_rel_preds,
                        output.dial_encoder_outputs.coh_rel_labels),
                    dial_encoder_coh_rel_f1=compute_f1(
                        output.dial_encoder_outputs.coh_rel_preds,
                        output.dial_encoder_outputs.coh_rel_labels),
                    dial_encoder_coh_rel_top_at_k=compute_hits_at_k(
                        output.dial_encoder_outputs.coh_rel_preds,
                        output.dial_encoder_outputs.coh_rel_labels,
                        k=5),
                    dial_encoder_link_pred_roc_auc=compute_roc_auc(
                        output.dial_encoder_outputs.link_prediction_logits.view(
                            -1).sigmoid(),
                        output.dial_encoder_outputs.link_prediction_labels),
                    dial_encoder_next_resp_type_direct_f1=compute_f1(
                        output.dial_encoder_outputs.next_resp_type_direct_preds,
                        output.dial_encoder_outputs.next_resp_type_direct_labels
                    ),
                    dial_encoder_next_resp_type_direct_top_at_k=
                    compute_hits_at_k(
                        output.dial_encoder_outputs.next_resp_type_direct_preds,
                        output.dial_encoder_outputs.
                        next_resp_type_direct_labels,
                        k=5),
                    dial_encoder_next_resp_type_seq_f1=compute_f1(
                        output.dial_encoder_outputs.next_resp_type_seq_preds,
                        output.dial_encoder_outputs.next_resp_type_seq_labels),
                    dial_encoder_next_resp_type_seq_top_at_k=compute_hits_at_k(
                        output.dial_encoder_outputs.next_resp_type_seq_preds,
                        output.dial_encoder_outputs.next_resp_type_seq_labels,
                        k=5),
                )

                self.logger.info(f'''Validation@bs{batch_idx}
                    [Coherence Relations] predicted probability: {coh_rel_preds.tolist()[-1]} |
                    [Coherence Relations] predicted labels: {[
                    label for prob, label in zip(coh_rel_preds[-1],
                                             COHERENCE_RELATIONS) if prob > 0.5
                ]}''')
                self.logger.info(f'''Validation@bs{batch_idx}
                    [Coherence Relations] ground-truth one-hots: {coh_rel_labels.tolist()[-1]} |
                    [Coherence Relations] ground-truth labels: {[
                    label for prob, label in zip(coh_rel_labels[-1],
                                             COHERENCE_RELATIONS) if prob == 1.0
                ]}''')
                self.logger.info(f'''Validation@bs{batch_idx}
                    [Dialogue Encoder Coherence Relations] predicted probability: {output.dial_encoder_outputs.coh_rel_preds.tolist()[-1]} |
                    [Dialogue Encoder Coherence Relations] predicted labels: {[
                    label for prob, label in zip(output.dial_encoder_outputs.coh_rel_preds[-1],
                                             COHERENCE_RELATIONS) if prob > 0.5
                ]}''')
                self.logger.info(f'''Validation@bs{batch_idx}
                    [Dialogue Encoder Coherence Relations] ground-truth one-hots: {output.dial_encoder_outputs.coh_rel_labels.tolist()[-1]} |
                    [Dialogue Encoder Coherence Relations] ground-truth labels: {[
                    label for prob, label in zip(output.dial_encoder_outputs.coh_rel_labels[-1],
                                                 COHERENCE_RELATIONS) if prob == 1.0
                ]}''')
                self.logger.info(f'''Validation@bs{batch_idx}
                    [Dialogue Encoder Next Response Types (Directly)] predicted probability: {output.dial_encoder_outputs.next_resp_type_direct_preds.tolist()[-1]} |
                    [Dialogue Encoder Next Response Types (Directly)] predicted labels: {[
                    label for prob, label in zip(output.dial_encoder_outputs.next_resp_type_direct_preds[-1],
                                                 COHERENCE_RELATIONS) if prob > 0.5
                ]}''')
                self.logger.info(f'''Validation@bs{batch_idx}
                    [Dialogue Encoder Next Response Types (Directly)] ground-truth one-hots: {output.dial_encoder_outputs.next_resp_type_direct_labels.tolist()[-1]} |
                    [Dialogue Encoder Next Response Types (Directly)] ground-truth labels: {[
                    label for prob, label in zip(output.dial_encoder_outputs.next_resp_type_direct_labels[-1],
                                                 COHERENCE_RELATIONS) if prob == 1.0
                ]}''')
                self.logger.info(f'''Validation@bs{batch_idx}
                    [Dialogue Encoder Next Response Types (Sequential)] predicted probability: {output.dial_encoder_outputs.next_resp_type_seq_preds.tolist()[-1]} |
                    [Dialogue Encoder Next Response Types (Sequential)] predicted labels: {[
                    label for prob, label in zip(output.dial_encoder_outputs.next_resp_type_seq_preds[-1],
                                                 COHERENCE_RELATIONS) if prob > 0.5
                ]}''')
                self.logger.info(f'''Validation@bs{batch_idx}
                    [Dialogue Encoder Next Response Types (Sequential)] ground-truth one-hots: {output.dial_encoder_outputs.next_resp_type_seq_labels.tolist()[-1]} |
                    [Dialogue Encoder Next Response Types (Sequential)] ground-truth labels: {[
                    label for prob, label in zip(output.dial_encoder_outputs.next_resp_type_seq_labels[-1],
                                                 COHERENCE_RELATIONS) if prob == 1.0
                ]}''')

        return metrics_accumulator.averages()

    def train(self) -> tuple[str, int]:
        model_path = ''
        endure_count = 0
        best_epoch = 0
        best_val_loss = float('inf')
        for epoch in tqdm(range(1, self.epochs + 1)):
            self.logger.info(f'epoch {epoch}')

            self.use_resp_type_prediction_in_training = scheduled_sampling_exp(
                epoch - 1, self.epochs)

            # Training
            train_output = self.training_step(self.train_data)
            self.logger.info(f'''[NLL] loss: {train_output.nll_loss:4.4f} |
                [Coherence Relations Classification] loss: {train_output.coh_rel_cls_loss:4.4f} on Training'''
                             )
            self.logger.info(
                f'''[Coherence Relations Classification] accuracy: {train_output.coh_rel_accuracy:4.4f} |
                [Coherence Relations Classification] f1: {train_output.coh_rel_f1:4.4f} |
                [Coherence Relations Classification] top@3: {train_output.coh_rel_top_at_k:4.4f} on Training'''
            )
            self.logger.info(
                f'''[Dialogue Encoder Coherence Relations Classification] loss: {train_output.dial_encoder_coh_rel_cls_loss:4.4f} |
                [Dialogue Encoder Link Prediction] loss: {train_output.dial_encoder_link_prediction_loss:4.4f} |
                [Dialogue Encoder Next Response Types (Directly)] loss: {train_output.dial_encoder_next_resp_type_direct_loss:4.4f} |
                [Dialogue Encoder Next Response Types (Sequential)] loss: {train_output.dial_encoder_next_resp_type_seq_loss:4.4f} on Training'''
            )
            self.logger.info(
                f'''[Dialogue Encoder Coherence Relations Classification] accuracy: {train_output.dial_encoder_coh_rel_accuracy:4.4f} |
                [Dialogue Encoder Coherence Relations Classification] f1: {train_output.dial_encoder_coh_rel_f1:4.4f} |
                [Dialogue Encoder Coherence Relations Classification] top@5: {train_output.dial_encoder_coh_rel_top_at_k:4.4f} |
                [Dialogue Encoder Link Prediction] roc-auc: {train_output.dial_encoder_link_pred_roc_auc:4.4f} |
                [Dialogue Encoder Next Response Types (Directly)] f1: {train_output.dial_encoder_next_resp_type_direct_f1:4.4f} |
                [Dialogue Encoder Next Response Types (Directly)] top@5: {train_output.dial_encoder_next_resp_type_direct_top_at_k:4.4f} |
                [Dialogue Encoder Next Response Types (Sequential)] f1: {train_output.dial_encoder_next_resp_type_seq_f1:4.4f} |
                [Dialogue Encoder Next Response Types (Sequential)] top@5: {train_output.dial_encoder_next_resp_type_seq_top_at_k:4.4f} on Training'''
            )

            # Validation
            valid_output = self.evaluate_step(self.valid_data)
            self.logger.info(f'''[NLL] loss: {valid_output.nll_loss:4.4f} |
                [Coherence Relations Classification] loss: {valid_output.coh_rel_cls_loss:4.4f} on Validation'''
                             )
            self.logger.info(
                f'''[Coherence Relations Classification] accuracy: {valid_output.coh_rel_accuracy:4.4f} |
                [Coherence Relations Classification] f1: {valid_output.coh_rel_f1:4.4f} |
                [Coherence Relations Classification] top@3: {valid_output.coh_rel_top_at_k:4.4f} on Validation'''
            )
            self.logger.info(
                f'''[Dialogue Encoder Coherence Relations Classification] loss: {valid_output.dial_encoder_coh_rel_cls_loss:4.4f} |
                [Dialogue Encoder Link Prediction] loss: {valid_output.dial_encoder_link_prediction_loss:4.4f} |
                [Dialogue Encoder Next Response Types (Directly)] loss: {valid_output.dial_encoder_next_resp_type_direct_loss:4.4f} |
                [Dialogue Encoder Next Response Types (Sequential)] loss: {valid_output.dial_encoder_next_resp_type_seq_loss:4.4f} on Training'''
            )
            self.logger.info(
                f'''[Dialogue Encoder Coherence Relations Classification] accuracy: {valid_output.dial_encoder_coh_rel_accuracy:4.4f} |
                [Dialogue Encoder Coherence Relations Classification] f1: {valid_output.dial_encoder_coh_rel_f1:4.4f} |
                [Dialogue Encoder Coherence Relations Classification] top@5: {valid_output.dial_encoder_coh_rel_top_at_k:4.4f} |
                [Dialogue Encoder Link Prediction] roc-auc: {valid_output.dial_encoder_link_pred_roc_auc:4.4f} |
                [Dialogue Encoder Next Response Types (Directly)] f1: {valid_output.dial_encoder_next_resp_type_direct_f1:4.4f} |
                [Dialogue Encoder Next Response Types (Directly)] top@5: {valid_output.dial_encoder_next_resp_type_direct_top_at_k:4.4f} |
                [Dialogue Encoder Next Response Types (Sequential)] f1: {valid_output.dial_encoder_next_resp_type_seq_f1:4.4f} |
                [Dialogue Encoder Next Response Types (Sequential)] top@5: {valid_output.dial_encoder_next_resp_type_seq_top_at_k:4.4f} on Training'''
            )

            self.scheduler.step(valid_output.total_loss)
            self.logger.info(
                f'Learning rate set to {self.scheduler.get_last_lr()[0]:2.8f}')

            if valid_output.total_loss < best_val_loss:
                endure_count = 0

                saved_file_prefix = f'{self.model_name}-{self.dataset_name}'
                saved_model_file = f'{saved_file_prefix}_model.pt'
                saved_state_dict = f'{saved_file_prefix}_state_dict.pt'
                model_path = os.path.join(self.checkpoint, saved_model_file)
                state_dict_path = os.path.join(self.checkpoint,
                                               saved_state_dict)

                if not self.distributed:
                    self.model.save_pretrained(self.checkpoint / 'model',
                                               safe_serialization=False)
                    save_model(self.model, model_path)
                    self.logger.info(
                        '(resume training ok) Save the best model' + model_path)
                    save_model(self.model.state_dict(), state_dict_path)
                    self.logger.info(
                        "(inference ok) Save the best model's state dict" +
                        state_dict_path)
                else:
                    self.model.save_pretrained(self.checkpoint / 'model',
                                               safe_serialization=False)
                    save_dist_model(self.accelerator, self.model,
                                    state_dict_path)
                    self.logger.info(
                        "(inference ok) Save the best model's state dict" +
                        state_dict_path)

                best_val_loss = valid_output.total_loss
                best_epoch = epoch
            else:
                endure_count += 1
                self.logger.info(f'Endured {endure_count} time(s)')
                if endure_count == self.endure_times:
                    self.logger.info(
                        'Cannot endure it anymore | Exiting from early stop')
                    break

        return model_path, best_epoch


TRAINER_CLASSES = {
    'dgat': {
        ModelTrainMode.FINETUNEING:
        DGatForCoherenceAwareDialogueEncodingTrainer,
        ModelTrainMode.PRETRAINING: DGatForPreTrainingTrainer,
    },
    'generator': {
        ModelTrainMode.FINETUNEING: PersonalizedDialogueGeneratorTrainer,
    },
    'end-to-end': {
        ModelTrainMode.FINETUNEING:
        CoherenceAwarePersonalizedDialogueGenerationSeq2SeqTrainer,
    }
}


def get_trainer_class(model_type: Literal['dgat', 'generator', 'end-to-end'],
                      train_mode: ModelTrainMode):
    return TRAINER_CLASSES[model_type][train_mode]
