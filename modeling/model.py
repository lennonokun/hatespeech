from torch.optim import AdamW

from adapters import AutoAdapterModel, LoRAConfig

from .multimodel import BaseMultiModel

class HateModule(BaseMultiModel):
  def __init__(self, config):
    super().__init__(config)

    self.model = AutoAdapterModel.from_pretrained(
      config["model"],
      low_cpu_mem_usage=True,
    )
    self.model.add_adapter("my_adapter", config=LoRAConfig(
      r=config["adapter_r"],
      # dropout=config["adapter_dropout"],
    ))
    self.model.set_active_adapters("my_adapter")
    self.model.train_adapter("my_adapter")
    
    norm_layers = [
      v for k,v in self.model.named_parameters()
      if ("layer.10" in k or "layer.11" in k) and "lora" in k
    ]
    print(sum(p.numel() for p in self.model.parameters() if p.requires_grad))
    self.set_norm_layers(norm_layers)

  def forward_base(self, batch):
    outputs = self.model(batch["tokens"], attention_mask=batch["mask"].float())
    return outputs.last_hidden_state

  def configure_optimizers(self):
    std_params = filter(lambda p: p.requires_grad, self.parameters())
    return AdamW(params= std_params, lr=self.config["learning_rate"])
