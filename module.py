import numpy as np

import torch
from torch import nn
from torch.optim import Adafactor, AdamW
from torch.optim.lr_scheduler import *

from torchmetrics import MetricCollection
from torchmetrics.classification import F1Score, Accuracy
from lightning import LightningModule, Trainer

# import adapters
# from adapters import LoReftConfig, DiReftConfig, LoRAConfig
from peft import get_peft_config, get_peft_model, LoraConfig
from transformers import AutoModel, BitsAndBytesConfig

class HateModule(LightningModule):
  def __init__(self, config):
    super().__init__()
    self.save_hyperparameters()
    self.config = config
    self.model = AutoModel.from_pretrained(
      config["model"],
      # quantization_config=BitsAndBytesConfig(
      #   load_in_4bit=True,
      #   bnb_4bit_quant_type="nf4",
      #   bnb_4bit_use_double_quant=True,
      #   bnb_4bit_compute_dtype=torch.bfloat16
      # ),
    )
    
    self.model = get_peft_model(self.model, LoraConfig(
      inference_mode=False,
      # inference_mode=False, target_modules=["attn.Wqkv", "attn.Wo"],
      r=16, lora_alpha=32
    ))

    # adapters.init(self.model)
    # self.model.add_adapter("adapter", DiReftConfig(r=16,dropout=0.1))
    # # self.model.add_adapter("adapter", LoRAConfig(
    # #   selfattn_lora=True, intermediate_lora=True, output_lora=True,
    # #   attn_matrices=["q", "k", "v"],
    # #   r=16, alpha=16, dropout=0.1,
    # # ))

    self.head = nn.Linear(self.model.config.hidden_size, config["num_labels"])

    # TODO: weights
    self.criterion = nn.CrossEntropyLoss()
    
    self.train_metrics = MetricCollection({
      "acc": Accuracy(
        task="multiclass",
        num_classes=config["num_labels"],
      ),
      "f1": F1Score(
        task="multiclass",
        num_classes=config["num_labels"],
        average="macro",
      ),
    }, prefix="train_")
    self.valid_metrics = self.train_metrics.clone(prefix="valid_")
    self.test_metrics = self.train_metrics.clone(prefix="test_")
   
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
    return self.head(pooled)

  def compute(self, tokens, mask, label):
    logits = self(tokens, mask)
    loss = self.criterion(logits, label)
    pred = torch.argmax(logits, axis=-1)
    return loss, pred
  
  def training_step(self, batch, batch_idx):
    tokens, mask, label = batch["tokens"], batch["mask"], batch["label"]
    loss, pred = self.compute(tokens, mask, label)
    metrics = self.train_metrics(pred, label)
    
    self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=True)
    self.log_dict(metrics, prog_bar=True, on_epoch=True, on_step=False)
    return loss
  
  def validation_step(self, batch, batch_idx):
    tokens, mask, label = batch["tokens"], batch["mask"], batch["label"]
    loss, pred = self.compute(tokens, mask, label)
    metrics = self.valid_metrics(pred, label)
    
    self.log("valid_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
    self.log_dict(metrics, prog_bar=True, on_epoch=True, on_step=False)
  
  def test_step(self, batch, batch_idx):
    tokens, mask, label = batch["tokens"], batch["mask"], batch["label"]
    loss, pred = self.compute(tokens, mask, label)
    metrics = self.test_metrics(pred, label)

    self.log_dict(metrics, prog_bar=True, on_epoch=True, on_step=False)
  
  def on_train_epoch_end(self):
    self.train_metrics.reset()
  
  def on_validation_epoch_end(self):
    self.valid_metrics.reset()
    
  def on_test_epoch_end(self):
    self.test_metrics.reset()
