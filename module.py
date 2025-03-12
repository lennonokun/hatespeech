# import numpy as np

import torch
from torch import nn
from torch.optim import AdamW
# from torch.optim.lr_scheduler import *

from torchmetrics import MetricCollection
from torchmetrics.classification import F1Score, Accuracy
from lightning import LightningModule

import adapters
from adapters import DiReftConfig
from transformers import AutoModel, QuantoConfig

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
    self.model.add_adapter("direft", DiReftConfig(r=8,dropout=0.2))
    self.model.set_active_adapters("direft")

    self.head_label = nn.Linear(self.model.config.hidden_size, config["num_labels"])
    self.head_target = nn.Linear(self.model.config.hidden_size, config["num_targets"])

    # TODO: weights / focal loss for targets and rationales
    self.criterion = nn.CrossEntropyLoss()
    
    self.splits = ["train", "valid", "test"]
    self.target_metrics = {split: MetricCollection({
      "acc": Accuracy(
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
   
  def configure_optimizers(self):
    optimizer = AdamW(
      filter(lambda p: p.requires_grad, self.parameters()),
      lr=self.config["learning_rate"],
      weight_decay=2e-2,
    )
    return optimizer

  def forward(self, tokens, mask):
    outputs = self.model(input_ids=tokens, attention_mask=mask)
    pooled = outputs.last_hidden_state[:, 0, :]
    return self.head_label(pooled), self.head_target(pooled)

  def compute(self, batch):
    tokens, mask, label, target = batch
    # tokens, mask, label = (torch.flatten(tensor, end_dim=1) for tensor in batch)
    logits_label, logits_target = self(tokens, mask)

    loss_label = self.criterion(logits_label, label)
    pred_label = torch.argmax(logits_label, dim=-1)
    hard_label = torch.argmax(label, dim=-1)

    loss_target = self.criterion(logits_target, target)
    pred_target = torch.ge(logits_target, 0)
    hard_target = torch.ge(target, 0.5)
    return (loss_label, pred_label, hard_label), (loss_target, pred_target, hard_target)

  def compute_step(self, batch, split):
    results_label, results_target = self.compute(batch)
    loss_label, pred_label, hard_label = results_label
    loss_target, pred_target, hard_target = results_target

    target_metrics = self.target_metrics[split](pred_target, hard_target)
    label_metrics = self.label_metrics[split](pred_label, hard_label)
    self.log_dict(target_metrics, prog_bar=True, on_epoch=True, on_step=(split == "train"))
    self.log_dict(label_metrics, prog_bar=True, on_epoch=True, on_step=(split == "train"))

    if split == "train":
      self.log(f"train_label_loss", loss_label, prog_bar=True, on_epoch=True, on_step=True)
      self.log(f"train_target_loss", loss_target, prog_bar=True, on_epoch=True, on_step=True)
      # TODO linear combination
      return self.config["label_loss_coef"] * loss_label + loss_target
    elif split == "valid":
      self.log(f"valid_label_loss", loss_label, prog_bar=True, on_epoch=True, on_step=False)
      self.log(f"valid_target_loss", loss_target, prog_bar=True, on_epoch=True, on_step=False)
  
  def training_step(self, batch, _):
    return self.compute_step(batch, "train")
  
  def validation_step(self, batch, _):
    return self.compute_step(batch, "valid")
  
  def test_step(self, batch, _):
    return self.compute_step(batch, "test")
  
  def on_split_epoch_end(self, split):
    self.label_metrics[split].reset()
    self.target_metrics[split].reset()
  
  def on_train_epoch_end(self):
    self.on_split_epoch_end("train")
  
  def on_validation_epoch_end(self):
    self.on_split_epoch_end("valid")
    
  def on_test_epoch_end(self):
    self.on_split_epoch_end("test")
