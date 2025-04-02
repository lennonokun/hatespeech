import numpy as np

import torch
from torch import nn
from torch.optim import AdamW
from torch.nn import functional as F
# from torch.optim.lr_scheduler import *

from torchmetrics import MetricCollection
from torchmetrics import classification as clf
from lightning import LightningModule

import adapters
from adapters import DiReftConfig
from transformers import AutoModel, QuantoConfig

from custom import MaskedBinaryAccuracy, MaskedBinaryF1, MyBCELoss

class HateModule(LightningModule):
  def __init__(self, config):
    super().__init__()
    self.save_hyperparameters()
    # self.automatic_optimization = False
    self.config = config
    self.model = AutoModel.from_pretrained(
      config["model"],
      low_cpu_mem_usage=True,
      quantization_config=QuantoConfig(
        quantization_dtype="nf4",
        # quantization_method="bitsandbytes",
        load_in_low_bit_precision=True,
        compute_dtype="bfloat16",
        quantize_embeddings=True,
      )
    )

    adapters.init(self.model)
    self.model.add_adapter("adapter", DiReftConfig(r=8))
    self.model.set_active_adapters("adapter")
    
    self.head_label = nn.Linear(self.model.config.hidden_size, config["num_labels"])
    self.head_target = nn.Linear(self.model.config.hidden_size, config["num_targets"])
    self.head_rationale = nn.Linear(self.model.config.hidden_size, 1)

    # TODO: weights / focal loss for targets and rationales
    self.label_loss = MyBCELoss()
    # self.target_loss = FocalLoss(alpha=0.03, gamma=1.5, multilabel=True)
    target_reduce_dim = 0 if config["multitask_targets"] else None
    self.target_loss = MyBCELoss(reduce_dim=target_reduce_dim)
    # self.rationale_loss = FocalLoss(alpha=0.3, gamma=2.0)
    self.rationale_loss = MyBCELoss()

    # each target is its own task
    # TODO differentiate num_tasks?
    if config["multitask_targets"]:
      self.num_tasks = config["num_targets"] + config["num_tasks"] - 1
      target_importances = 5e0 * np.ones(config["num_targets"]) / config["num_targets"]
      task_importances = np.concat(([1], target_importances, [1]), axis=0)
    else:
      self.num_tasks = config["num_tasks"]
      task_importances = np.ones(self.num_tasks)

    self.task_importances = torch.Tensor(task_importances).cuda()
    self.task_weights = nn.Parameter(torch.zeros(self.num_tasks)) # log space
    self.task_regularization = None
    
    self.splits = ["train", "valid", "test"]
    self.target_metrics = {split: MetricCollection({
      "acc": clf.MultilabelAccuracy(
        num_labels=config["num_targets"],
        average="micro",
      ), "f1": clf.MultilabelF1Score(
        num_labels=config["num_targets"],
        average="micro",
      )
    }, prefix=f"{split}_target_").cuda() for split in self.splits}
    self.label_metrics = {split: MetricCollection({
      "acc": clf.MulticlassAccuracy(
        num_classes=config["num_labels"],
        average="macro",
      ), "f1": clf.MulticlassF1Score(
        num_classes=config["num_labels"],
        average="macro",
      ),
    }, prefix=f"{split}_label_").cuda() for split in self.splits}
    self.rationale_metrics = {split: MetricCollection({
      "acc": MaskedBinaryAccuracy(),
      "f1": MaskedBinaryF1(),
    }, prefix=f"{split}_rationale_").cuda() for split in self.splits}

    self.target_test_metrics = MetricCollection({
      "precision": clf.MultilabelPrecision(
        num_labels=config["num_targets"],
        average="macro",
      ), "recall": clf.MultilabelRecall(
        num_labels=config["num_targets"],
        average="macro",
      ),
    }, prefix="test_target_").cuda()
    self.target_test_f1_metric = clf.MultilabelF1Score(
      num_labels=config["num_targets"],
      average="none",
    )
   
  def configure_optimizers(self):
    std_params = filter(lambda p: p is not self.task_weights and p.requires_grad, self.parameters())
    return AdamW([
      {"params": std_params, "lr": self.config["learning_rate"]},
      {"params": [self.task_weights], "lr": self.config["learning_rate"] * 10},
    ])

  def on_fit_start(self):
    self.label_loss.set_freq(self.trainer.datamodule.stats["label_freq"])
    self.target_loss.set_freq(self.trainer.datamodule.stats["target_freq"])
    self.rationale_loss.set_freq(self.trainer.datamodule.stats["rationale_freq"])

  def on_test_start(self):
    self.on_fit_start()

  def forward(self, tokens, mask):
    outputs = self.model(input_ids=tokens, attention_mask=mask)
    nonpooled = outputs.last_hidden_state
    pooled = outputs.last_hidden_state[:, 0, :]
    logits_label = self.head_label(pooled)
    logits_target = self.head_target(pooled)
    logits_rationale = self.head_rationale(nonpooled).squeeze(-1)
    return logits_label, logits_target, logits_rationale

  @torch.no_grad()
  def predict(self, tokens, mask):
    tokens, mask = tuple(torch.from_numpy(arr).to(self.model.device) for arr in [tokens, mask])
    logits_label, logits_target, logits_rationale = self(tokens, mask)
    prob_label = F.softmax(logits_label, dim=1)
    prob_target = F.sigmoid(logits_target)
    prob_rationale = F.sigmoid(logits_rationale)
    return tuple(tensor.cpu().numpy() for tensor in [
      prob_label, prob_target, prob_rationale
    ])
  
  def compute(self, batch):
    tokens, mask, label, target, rationale = batch
    logits_label, logits_target, logits_rationale = self(tokens, mask)

    loss_label = self.label_loss(logits_label, label)
    pred_label = torch.argmax(logits_label, dim=-1)
    hard_label = torch.argmax(label, dim=-1)
    results_label = (loss_label, pred_label, hard_label)

    loss_target = self.target_loss(logits_target, target)
    pred_target = torch.ge(logits_target, 0)
    hard_target = torch.ge(target, 0.5)
    results_target = (loss_target, pred_target, hard_target)

    # only check given rationales
    # todo weigh by number of valid annotations?
    mask_rationale = mask & label[:, 1].gt(0)[:, None]
    loss_rationale = self.rationale_loss(logits_rationale, rationale, mask_rationale)
    pred_rationale = torch.ge(logits_rationale, 0)
    hard_rationale = torch.ge(rationale, 0.5)
    results_rationale = (loss_rationale, pred_rationale, hard_rationale, mask_rationale)
    
    return results_label, results_target, results_rationale

  def compute_step(self, batch, split):
    results_label, results_target, results_rationale = self.compute(batch)
    loss_label, pred_label, hard_label = results_label
    loss_target, pred_target, hard_target = results_target
    loss_rationale, pred_rationale, hard_rationale, mask_rationale = results_rationale

    target_metrics = self.target_metrics[split](pred_target, hard_target)
    label_metrics = self.label_metrics[split](pred_label, hard_label)
    rationale_metrics = self.rationale_metrics[split](pred_rationale, hard_rationale, mask_rationale)
    target_test_metrics = self.target_test_metrics(pred_target, hard_target)
    target_test_f1 = self.target_test_f1_metric(pred_target, hard_target)

    log_metrics_on_step = split == "train" and self.config["logging"] != "terminal"

    # TODO for now dont clog up terminal
    if split != "train":
      self.log_dict(target_metrics, prog_bar=True, on_epoch=True, on_step=log_metrics_on_step)
      self.log_dict(label_metrics, prog_bar=True, on_epoch=True, on_step=log_metrics_on_step)
      self.log_dict(rationale_metrics, prog_bar=True, on_epoch=True, on_step=log_metrics_on_step)

    if self.config["multitask_targets"]:
      losses = torch.cat([loss_label.unsqueeze(-1), loss_target, loss_rationale.unsqueeze(-1)], dim=0)
    else:
      losses = torch.stack([loss_label, loss_target, loss_rationale])

    if self.task_regularization is None:
      self.task_regularization = 1 / (losses.detach() + 1e-8)
    losses *= self.task_importances
    losses *= self.task_regularization
      
    loss = 0.5 * (losses / (self.task_weights.exp() ** 2)).sum() + self.task_weights.sum()

    if split == "train":
      log_loss_on_step = self.config["logging"] != "terminal"
      self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=log_loss_on_step)
      # self.log(f"train_label_loss", loss_label, prog_bar=True, on_epoch=True, on_step=log_loss_on_step)
      # self.log(f"train_target_loss", loss_target, prog_bar=True, on_epoch=True, on_step=log_loss_on_step)
      # self.log(f"train_rationale_loss", loss_rationale, prog_bar=True, on_epoch=True, on_step=log_loss_on_step)
      return loss
    elif split == "valid":
      self.log("valid_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
      # self.log(f"valid_label_loss", loss_label, prog_bar=True, on_epoch=True, on_step=False)
      # self.log(f"valid_target_loss", loss_target, prog_bar=True, on_epoch=True, on_step=False)
      # self.log(f"valid_rationale_loss", loss_rationale, prog_bar=True, on_epoch=True, on_step=False)
    elif split == "test":
      self.log_dict(target_test_metrics, prog_bar=False, on_epoch=True, on_step=False)
      for f1, label in zip(target_test_f1, self.config["targets"]):
        self.log(f"test_target_{label.lower()}_f1", f1, prog_bar=False, on_epoch=True, on_step=False)

  def training_step(self, batch):
    return self.compute_step(batch, "train")
 
  def validation_step(self, batch, _):
    return self.compute_step(batch, "valid")
  
  def test_step(self, batch, _):
    return self.compute_step(batch, "test")
  
  def on_split_epoch_end(self, split):
    self.label_metrics[split].reset()
    self.target_metrics[split].reset()
    self.rationale_metrics[split].reset()
  
  def on_train_epoch_end(self):
    self.on_split_epoch_end("train")
  
  def on_validation_epoch_end(self):
    self.on_split_epoch_end("valid")
    
  def on_test_epoch_end(self):
    self.on_split_epoch_end("test")
