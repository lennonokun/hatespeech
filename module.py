# import numpy as np

import torch
from torch import nn
from torch.optim import AdamW
from torch.nn import functional as F
# from torch.optim.lr_scheduler import *

from torchmetrics import Metric
from torchmetrics import MetricCollection
from torchmetrics.classification import F1Score, Accuracy
from lightning import LightningModule

import adapters
from adapters import DiReftConfig, LoRAConfig, LoReftConfig
from transformers import AutoModel, QuantoConfig

class MaskedBinaryAccuracy(Metric):
  def __init__(self, dist_sync_on_step=False):
    super().__init__(dist_sync_on_step=dist_sync_on_step)
    self.add_state("correct", default=torch.tensor(0.), dist_reduce_fx="sum")
    self.add_state("incorrect", default=torch.tensor(0.), dist_reduce_fx="sum")

  def update(self, preds: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor):
    preds, labels, mask = preds.bool(), labels.bool(), mask.bool()
    self.correct += ((preds == labels) & mask).sum().float()
    self.incorrect += ((preds != labels) & mask).sum().float()

  def compute(self):
    return self.correct / (self.correct + self.incorrect)

class MaskedBinaryF1(Metric):
  def __init__(self, dist_sync_on_step=False):
    super().__init__(dist_sync_on_step=dist_sync_on_step)
    self.add_state("true_positives", default=torch.tensor(0.), dist_reduce_fx="sum")
    self.add_state("true_negatives", default=torch.tensor(0.), dist_reduce_fx="sum")
    self.add_state("false_positives", default=torch.tensor(0.), dist_reduce_fx="sum")
    self.add_state("false_negatives", default=torch.tensor(0.), dist_reduce_fx="sum")

  def update(self, preds: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor):
    preds, labels, mask = preds.bool(), labels.bool(), mask.bool()

    self.true_positives += (preds & labels & mask).sum().float()
    self.true_negatives += (~preds & ~labels & mask).sum().float()
    self.false_positives += (preds & ~labels & mask).sum().float()
    self.false_negatives += (~preds & labels & mask).sum().float()

  def compute(self):
    pos_precision = self.true_positives / (self.true_positives + self.false_positives + 1e-8)
    pos_recall = self.true_positives / (self.true_positives + self.false_negatives + 1e-8)
    pos_f1 = 2 * pos_precision * pos_recall / (pos_precision + pos_recall + 1e-8)
    neg_precision = self.true_negatives / (self.true_negatives + self.false_negatives + 1e-8)
    neg_recall = self.true_negatives / (self.true_negatives + self.false_positives + 1e-8)
    neg_f1 = 2 * neg_precision * neg_recall / (neg_precision + neg_recall + 1e-8)
    return (pos_f1 + neg_f1) / 2

class MaskedFocalLoss(nn.Module):
  def __init__(self, alpha, gamma, epsilon):
    super().__init__()
    self.alpha = alpha
    self.gamma = gamma
    self.epsilon = epsilon

  def forward(self, logits, labels, masks):
    labels = labels * (1 - self.epsilon) + (1 - labels) * self.epsilon
    
    probs = torch.sigmoid(logits)
    probs_t = probs * labels + (1 - probs) * (1 - labels)
    alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
    bce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    loss = alpha_t * ((1 - probs_t) ** self.gamma) * bce_loss
    return (loss * masks).sum() / (masks.sum() + 1e-8)

class HateModule(LightningModule):
  def __init__(self, config):
    super().__init__()
    self.save_hyperparameters()
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
    # self.model.add_adapter("adapter", DiReftConfig(r=8, dropout=0.1))
    # self.model.add_adapter("adapter", LoRAConfig(r=8, alpha=32))
    self.model.add_adapter("adapter", LoReftConfig(r=8))

    self.model.set_active_adapters("adapter")

    self.head_label = nn.Linear(self.model.config.hidden_size, config["num_labels"])
    self.head_target = nn.Linear(self.model.config.hidden_size, config["num_targets"])
    self.head_rationale = nn.Linear(self.model.config.hidden_size, 1)

    # TODO: weights / focal loss for targets and rationales
    self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.05)
    # self.bce_loss = nn.BCEWithLogitsLoss()
    self.focal_loss = MaskedFocalLoss(alpha=0.2, gamma=3.0, epsilon=0.05)
    # log space
    self.sigma = nn.Parameter(torch.zeros(3))
    
    self.splits = ["train", "valid", "test"]
    self.target_metrics = {split: MetricCollection({
      "acc": Accuracy(
        task="multilabel",
        num_labels=config["num_targets"],
        average="macro",
      ), "f1": F1Score(
        task="multilabel",
        num_labels=config["num_targets"],
        average="macro",
      )
    }, prefix=f"{split}_target_").cuda() for split in self.splits}
    self.label_metrics = {split: MetricCollection({
      "acc": Accuracy(
        task="multiclass",
        num_classes=config["num_labels"],
        average="macro",
      ), "f1": F1Score(
        task="multiclass",
        num_classes=config["num_labels"],
        average="macro",
      ),
    }, prefix=f"{split}_label_").cuda() for split in self.splits}
    self.rationale_metrics = {split: MetricCollection({
      "acc": MaskedBinaryAccuracy(),
      "f1": MaskedBinaryF1(),
    }, prefix=f"{split}_rationale_").cuda() for split in self.splits}

   
  def configure_optimizers(self):
    std_params = filter(lambda p: p is not self.sigma and p.requires_grad, self.parameters())
    optimizer = AdamW([
      {"params": std_params, "lr": self.config["learning_rate"]},
      {"params": self.sigma, "lr": self.config["learning_rate"] * 100},
    ])
    return optimizer

  def forward(self, tokens, mask):
    outputs = self.model(input_ids=tokens, attention_mask=mask)
    nonpooled = outputs[0] # todo
    pooled = outputs.last_hidden_state.mean(dim=1)
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

    loss_label = self.ce_loss(logits_label, label)
    pred_label = torch.argmax(logits_label, dim=-1)
    hard_label = torch.argmax(label, dim=-1)
    results_label = (loss_label, pred_label, hard_label)

    loss_target = self.focal_loss(logits_target, target, torch.ones_like(target))
    pred_target = torch.ge(logits_target, 0)
    hard_target = torch.ge(target, 0.5)
    results_target = (loss_target, pred_target, hard_target)

    # only check given rationales
    # todo weigh by number of valid annotations?
    mask_rationale = mask & torch.reshape(label[:, 1].ge(0), (-1, 1))
    loss_rationale = self.focal_loss(logits_rationale, rationale, mask_rationale)
    pred_rationale = torch.ge(logits_rationale, 0)
    hard_rationale = torch.ge(rationale, 0.5)
    results_rationale = (loss_rationale, pred_rationale, hard_rationale)
    
    return results_label, results_target, results_rationale

  def compute_step(self, batch, split):
    mask = batch[1]
    results_label, results_target, results_rationale = self.compute(batch)
    loss_label, pred_label, hard_label = results_label
    loss_target, pred_target, hard_target = results_target
    loss_rationale, pred_rationale, hard_rationale = results_rationale

    target_metrics = self.target_metrics[split](pred_target, hard_target)
    label_metrics = self.label_metrics[split](pred_label, hard_label)
    rationale_metrics = self.rationale_metrics[split](pred_rationale, hard_rationale, mask)
    log_metrics_on_step = split == "train" and self.config["logging"] != "terminal"
    self.log_dict(target_metrics, prog_bar=True, on_epoch=True, on_step=log_metrics_on_step)
    self.log_dict(label_metrics, prog_bar=True, on_epoch=True, on_step=log_metrics_on_step)
    self.log_dict(rationale_metrics, prog_bar=True, on_epoch=True, on_step=log_metrics_on_step)

    loss = torch.stack([loss_label, loss_target, loss_rationale])
    weights = self.sigma.exp()
    loss = 0.5 * (loss / weights**2).sum() + weights.prod().log()
    if split == "train":
      log_loss_on_step = self.config["logging"] != "terminal"
      # self.log(f"train_label_loss", loss_label, prog_bar=True, on_epoch=True, on_step=log_loss_on_step)
      # self.log(f"train_target_loss", loss_target, prog_bar=True, on_epoch=True, on_step=log_loss_on_step)
      # self.log(f"train_rationale_loss", loss_rationale, prog_bar=True, on_epoch=True, on_step=log_loss_on_step)
      self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=log_loss_on_step)
      return loss
    elif split == "valid":
      self.log("valid_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
      # self.log(f"valid_label_loss", loss_label, prog_bar=True, on_epoch=True, on_step=False)
      # self.log(f"valid_target_loss", loss_target, prog_bar=True, on_epoch=True, on_step=False)
      # self.log(f"valid_rationale_loss", loss_rationale, prog_bar=True, on_epoch=True, on_step=False)
    
  def training_step(self, batch, _):
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
