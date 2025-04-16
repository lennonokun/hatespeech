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
from transformers import AutoModel 

from .custom import MaskedBinaryAccuracy, MaskedBinaryF1, MyBCELoss

class HateModule(LightningModule):
  def __init__(self, cfg):
    super().__init__()
    self.save_hyperparameters()
    # self.automatic_optimization = False
    self.cfg = cfg

    self.model = AutoModel.from_pretrained(
      cfg["model"],
      low_cpu_mem_usage=True,
    )

    adapters.init(self.model)
    self.model.add_adapter("adapter", DiReftConfig(r=8))
    self.model.set_active_adapters("adapter")
    
    self.head_label = nn.Linear(self.model.config.hidden_size, cfg["num_label"])
    self.head_target = nn.Linear(self.model.config.hidden_size, cfg["num_target"])
    self.head_rationale = nn.Linear(self.model.config.hidden_size, 1)

    # TODO: weights / focal loss for targets and rationales
    self.label_loss = MyBCELoss(freq=cfg["stats"]["label_freqs"])
    # self.target_loss = FocalLoss(alpha=0.03, gamma=1.5, multilabel=True)
    target_reduce_dim = 0 if cfg["multitask_targets"] else None
    self.target_loss = MyBCELoss(freq=cfg["stats"]["target_freqs"], reduce_dim=target_reduce_dim)
    # self.rationale_loss = FocalLoss(alpha=0.3, gamma=2.0)
    self.rationale_loss = MyBCELoss(freq=cfg["stats"]["rationale_freq"])

    # each target is its own task
    # TODO differentiate num_tasks?
    if cfg["multitask_targets"]:
      self.num_tasks = cfg["num_target"] + cfg["num_tasks"] - 1
      target_importances = np.full(cfg["num_target"], 5e0 / cfg["num_target"])
      task_importances = np.concatenate(([1], target_importances, [1]), axis=0)
    else:
      self.num_tasks = cfg["num_tasks"]
      task_importances = np.ones(self.num_tasks)
    task_importances /= self.num_tasks

    self.task_importances = torch.Tensor(task_importances).cuda()
    if cfg["mtl_weighing"] == "uw":
      self.task_weights = nn.Parameter(torch.zeros(self.num_tasks)) # log space
    elif cfg["mtl_weighing"] == "dwa":
      self.task_weights = torch.ones(self.num_tasks, requires_grad=False)

    self.initial_loss_history = []
    self.dwa_loss_history = []
    self.task_norm = None
    
    self.splits = ["train", "valid", "test"]
    self.target_metrics = {split: MetricCollection({
      # "acc": clf.MultilabelAccuracy(
      #   num_labels=cfg["num_target"],
      #   average="micro",
      # ),
      "f1": clf.MultilabelF1Score(
        num_labels=cfg["num_target"],
        average="micro",
      )
    }, prefix=f"{split}_target_").cuda() for split in self.splits}
    self.label_metrics = {split: MetricCollection({
      # "acc": clf.MulticlassAccuracy(
      #   num_classes=cfg["num_label"],
      #   average="macro",
      # ),
      "f1": clf.MulticlassF1Score(
        num_classes=cfg["num_label"],
        average="macro",
      ),
    }, prefix=f"{split}_label_").cuda() for split in self.splits}
    self.rationale_metrics = {split: MetricCollection({
      # "acc": MaskedBinaryAccuracy(),
      "f1": MaskedBinaryF1(),
    }, prefix=f"{split}_rationale_").cuda() for split in self.splits}
   
  def configure_optimizers(self):
    std_params = filter(lambda p: p is not self.task_weights and p.requires_grad, self.parameters())
    arg = [{"params": std_params, "lr": self.cfg["learning_rate"]}]
    if self.cfg["mtl_weighing"] == "uw":
      arg.append({"params": [self.task_weights], "lr": self.cfg["learning_rate"] * 10})
    return AdamW(arg)

  def forward(self, embeddings, mask):
    outputs = self.model(inputs_embeds=embeddings, attention_mask=mask.float())
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

  def get_embeddings(self, tokens):
    return self.model.embeddings(tokens)

  def mtl_loss(self, loss_label, loss_target, loss_rationale, virtual):
    if self.cfg["multitask_targets"]:
      losses = torch.cat([
        loss_label.unsqueeze(-1),
        loss_target,
        loss_rationale.unsqueeze(-1)
      ], dim=0)
    else:
      losses = torch.stack([loss_label, loss_target, loss_rationale])

    if not virtual and self.cfg["mtl_norm_initial"]:
      if len(self.initial_loss_history) < self.cfg["mtl_norm_length"]:
        self.initial_loss_history.append(losses.detach())
        if len(self.initial_loss_history) == self.cfg["mtl_norm_length"]:
          mean_losses = torch.mean(torch.stack(self.initial_loss_history), dim=0) 
          self.task_norm = 1 / (mean_losses + 1e-8)

    if self.cfg["mtl_weighing"] == "uw":
      if self.task_norm is not None:
        losses *= self.task_norm
      losses *= self.task_importances
      loss = 0.5 * (losses / (self.task_weights.exp() ** 2)).sum() + self.task_weights.sum()
      return loss
    elif self.cfg["mtl_weighing"] == "dwa":
      if self.task_norm is not None:
        losses *= self.task_norm
        
      if len(self.dwa_loss_history) == 2:
        ratios = self.dwa_loss_history[0] / (self.dwa_loss_history[1] + 1e-8)
      else:
        ratios = torch.ones(self.num_tasks).cuda()

      loss = losses @ (F.softmax(ratios / self.cfg["mtl_dwa_T"], dim=0) * self.task_importances)
      if len(self.dwa_loss_history) == 2:
        self.dwa_loss_history.pop(0)
      self.dwa_loss_history.append(losses.detach())

      return loss
    else:
      return losses.sum()
    # TODO return all results
  
  def compute(self, embeddings, mask, annotations, virtual=False):
    label, target, rationale = annotations
    logits_label, logits_target, logits_rationale = self(embeddings, mask)

    pred_label = torch.argmax(logits_label, dim=-1)
    hard_label = torch.argmax(label, dim=-1)
    results_label = (pred_label, hard_label)

    pred_target = torch.ge(logits_target, 0)
    hard_target = torch.ge(target, 0.5)
    results_target = (pred_target, hard_target)

    mask_rationale = mask & label[:, 1].gt(0)[:, None]
    pred_rationale = torch.ge(logits_rationale, 0)
    hard_rationale = torch.ge(rationale, 0.5)
    results_rationale = (pred_rationale, hard_rationale, mask_rationale)

    loss_label = self.label_loss(logits_label, label)
    loss_target = self.target_loss(logits_target, target)
    loss_rationale = self.rationale_loss(logits_rationale, rationale, mask_rationale)
    loss = self.mtl_loss(loss_label, loss_target, loss_rationale, virtual)
    
    return results_label, results_target, results_rationale, loss
    
  def virtual_adversary(self, tokens, mask, annotations):
    embeddings = self.model.embeddings(tokens)
    embeddings.requires_grad_(True)

    results = self.compute(embeddings, mask, annotations, virtual=True)
    loss = results[-1]

    grad = torch.autograd.grad(loss, embeddings)[0].detach()
    coef = self.cfg["vat_epsilon"] * (1 - F.sigmoid(grad))
    perturbation = coef * grad / (torch.norm(grad) + 1e-8)
    # TODO no perturbation on special tokens?

    return (embeddings + perturbation).detach()
  
  def compute_step(self, batch, split):
    b1, b2 = batch
    annotations = (b1["label"], b1["target"], b1["rationale"])
    b1_size = b1["label"].shape[0]

    if split == "train" and self.cfg["vat_epsilon"] > 0.0:
      embeddings = self.virtual_adversary(b1["tokens"], b1["mask"], b1["annotations"])
    else:
      embeddings = self.get_embeddings(b1["tokens"])

    results = self.compute(embeddings, b1["mask"], annotations)
    results_label, results_target, results_rationale, loss = results

    label_metrics = self.label_metrics[split](*results_label)
    target_metrics = self.target_metrics[split](*results_target)
    rationale_metrics = self.rationale_metrics[split](*results_rationale)

    log_loss_on_step = self.cfg["logging"] != "terminal"
    log_metrics_on_step = split == "train" and log_loss_on_step

    # TODO for now dont clog up terminal
    if split != "train":
      self.log_dict(target_metrics, prog_bar=True, on_epoch=True, batch_size=b1_size, on_step=log_metrics_on_step)
      self.log_dict(label_metrics, prog_bar=True, on_epoch=True, batch_size=b1_size, on_step=log_metrics_on_step)
      self.log_dict(rationale_metrics, prog_bar=True, on_epoch=True, batch_size=b1_size, on_step=log_metrics_on_step)

    if split == "train":
      self.log("train_loss", loss, prog_bar=True, on_epoch=True, batch_size=b1_size, on_step=log_loss_on_step)
      return loss
    elif split == "valid":
      self.log("valid_loss", loss, prog_bar=True, on_epoch=True, batch_size=b1_size, on_step=False)

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
