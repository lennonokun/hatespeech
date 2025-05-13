import numpy as np
import torch
from torch import nn

from transformers import AutoModel, BitsAndBytesConfig
import adapters
from bitsandbytes.optim import *
from torch.optim.lr_scheduler import *

from lightning import LightningModule

from .mtl_loss import construct_mtl_loss
from .tasks import construct_tasks
from .custom import MultiLR
  
class BaseModel(LightningModule):
  def __init__(self, config, model_config=None):
    super().__init__()
    self.save_hyperparameters()

    self.lr = config["learning_rate"]
    self.config = config
    self.model = self.create_adapter_model(config, model_config)
    self.tasks = nn.ModuleDict(construct_tasks(config))
    self.mtl_loss = construct_mtl_loss(config, self.tasks)
    # SHOULD BE SET IN SUBCLASS __init__ OR NO GRAD NORM
    self.norm_layers = None

  def create_adapter_model(self, config, model_config):
    bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_use_double_quant=True,
      bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModel.from_pretrained(
      config["model"],
      config=model_config,
      low_cpu_mem_usage=True,
      device_map="auto",
      torch_dtype=torch.bfloat16,
      quantization_config=bnb_config if config["quantize"] else None,
    )
    adapters.init(model)
    return model

  def vis_params(self):
    params = list(self.model.named_parameters())
    max_name_len = max(len(k) for k,_ in params)
    grad_params = [(k, v) for k,v in params if v.requires_grad]
    no_grad_params = [(k, v) for k,v in params if not v.requires_grad]
    print(f"grad params:")
    for k, v in grad_params:
      print(f"  {k:{max_name_len}}  {str(v.dtype):15s}  {str(v.shape):15s}")
    print(f"no grad params:")
    for k, v in no_grad_params:
      print(f"  {k:{max_name_len}}  {str(v.dtype):15s}  {str(v.shape):15s}")

  def adjust_dtypes(self):
    for name, param in self.model.named_parameters():
      if "adapter" in name or "lora" in name:
        param.data = param.data.to(torch.bfloat16)

  def save(self, dest):
    if dest is not None:
      self.model.save_all_adapters(dest)

  def forward_base(self, batch):
    return self.model(
      input_ids=batch["tokens"],
      attention_mask=batch["mask"].bfloat16()
    ).last_hidden_state.float()

  def forward(self, batches, split):
    metricss, losses, sizes = {}, {}, {}
    for set_name in self.config["flat_datasets"]:
      batch = batches[set_name]
      hidden = self.forward_base(batch)
      for task_name in self.config["active_tasks"][set_name]:
        metricss[task_name], losses[task_name] = self.tasks[task_name].compute(hidden, batch, split)
        sizes[task_name] = batch["size"]

    return metricss, losses, sizes

  def loss_weights(self, sizes):
    loss_counts = torch.Tensor(np.concatenate([
      np.full(self.tasks[name].loss_dim, sizes[name])
      for name in self.config["melt_tasks"]
    ], axis=0)).to(self.model.device)
    return loss_counts / loss_counts.mean()
    
  def split_step(self, batches, split):
    if any(x["size"] == 0 for x in batches.values()):
      return None
    
    metricss, losses, sizes = self.forward(batches, split)
    weights = self.loss_weights(sizes)
      
    log_kwargs = {"prog_bar": split == "test", "on_epoch": True, "on_step": False}

    # log metrics
    for name, metrics in metricss.items():
      self.log_dict(metrics, batch_size=sizes[name], **log_kwargs)

    # log losses
    if split != "test":
      for name, loss in losses.items():
        loss = loss.float().cpu().detach().numpy()
        for idx, val in np.ndenumerate(loss):
          str_idx = "_".join(str(x) for x in idx)
          prefix = "_" if str_idx else ""
          log_name = f"{split}_{name}_loss{prefix}{str_idx}"
          self.log(log_name, val, batch_size=sizes[name], **log_kwargs)

    return self.mtl_loss(losses, self.norm_layers, weights, split)

  def configure_optimizers(self): # pyright: ignore
    head_params = self.tasks.parameters()
    model_params = filter(lambda p: p.requires_grad, self.model.parameters())
    optimizer = PagedAdamW32bit([
      {"params": head_params, "lr": self.lr},
      {"params": model_params, "lr": 5*self.lr}
    ])

    def head_scheduler(optimizer):
      return CosineAnnealingLR(
        optimizer,
        T_max=1,
        eta_min=0.05*self.lr,
      )

    def model_scheduler(optimizer):
      return SequentialLR(
        optimizer,
        schedulers=[
          LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=self.config["model_warmup"],
          ), CosineAnnealingLR(
            optimizer,
            T_max=1,
            eta_min=0.01*self.lr,
          )
        ],
        milestones=[self.config["model_warmup"]]
      )

    scheduler = MultiLR(optimizer, [head_scheduler, model_scheduler])
    scheduler_dict = {"scheduler": scheduler, "interval": "epoch"}
    return [optimizer], [scheduler_dict]

  def state_dict(self, *args, **kwargs):
    state = super().state_dict(*args, **kwargs)
    quant_ends = ('absmax', 'scale', 'quant_map', 'bitsandbytes__nf4')
    return {k: v for k, v in state.items() if not k.endswith(quant_ends)}

  def dequantize(self):
    self.model = self.model.to("cuda")
    self.model.dequantize()
    
  def training_step(self, batch):
    return self.split_step(batch, "train")
 
  def validation_step(self, batch, _):
    return self.split_step(batch, "valid")
  
  def test_step(self, batch, _):
    return self.split_step(batch, "test")

  def on_split_epoch_end(self, split):
    for task in self.tasks.values():
      task.metrics[split].reset()
    self.mtl_loss.reset_norms()

  def on_train_epoch_end(self):
    self.on_split_epoch_end("train")
  
  def on_validation_epoch_end(self):
    self.on_split_epoch_end("valid")
    
  def on_test_epoch_end(self):
    self.on_split_epoch_end("test")
