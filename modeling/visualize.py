import torch
import numpy as np

from transformers import AutoTokenizer

from .module import HateModule
from .datamodule import HateDatamodule

class HateVisualizer:
  def __init__(self, config):
    self.config = config
    self.tokenizer = AutoTokenizer.from_pretrained(config["model"])
    self.module = HateModule.load_from_checkpoint(
      config["best_model"],
      map_location=torch.device("cuda"),
    )
    self.module.eval()
    self.datamodule = HateDatamodule(config)
    self.datamodule.setup("visualize")
  
  def tokenize(self, texts):
    tokenized = self.tokenizer(
      texts,
      padding="max_length",
      truncation=True,
      max_length=self.config["max_length"],
      return_offsets_mapping=True,
    )

    # find token labels
    tokens = np.array(tokenized["input_ids"], dtype=np.int32)
    mask = np.array(tokenized["attention_mask"], dtype=np.int32)
    offsets = np.array(tokenized["offset_mapping"], dtype=np.int32)
    return tokens, mask, offsets

  def evaluate_single(self, text):
    tokens, mask, offsets = self.tokenize([text])
    results = self.module.predict(tokens, mask)
    prob_label, prob_target, prob_rationale = tuple(arr.squeeze(0) for arr in results)

    pred_label_idx = np.argmax(prob_label)
    preds_target_idx = np.argwhere(prob_target > 0.5).squeeze(1)

    eval_prob_label = prob_label[pred_label_idx]
    eval_probs_target = prob_target[preds_target_idx]
    pred_label = self.config["cats_label"][pred_label_idx]
    preds_target = [self.config["cats_target"][idx] for idx in preds_target_idx]

    vis_rationale = self.visualize_rationale(prob_rationale > 0.5, offsets[0])

    print(f"       {vis_rationale}")
    print(f"{pred_label=}")
    print(f"{eval_prob_label=}")
    print(f"{preds_target=}")
    print(f"{eval_probs_target=}")

  # todo add rationales
  # todo rationales mask
  def evaluate(self, text_info, label_info):
    texts, tokens, mask, offsets = text_info
    label, target, rationale, spans = label_info

    prob_label, prob_target, prob_rationale = self.module.predict(tokens, mask)

    trues_label_idx = np.argmax(label, axis=1)
    preds_label_idx = np.argmax(prob_label, axis=1)
    trues_target_idx = np.argwhere(target > 0.5)
    preds_target_idx = np.argwhere(prob_target > 0.5)
    trues_rationale = rationale > (0.5 - 1e-7)
    preds_rationale = prob_rationale > (0.5 - 1e-7)

    for i in range(len(texts)):
      true_label = self.config["cats_label"][trues_label_idx[i]]
      pred_label = self.config["cats_label"][preds_label_idx[i]]

      true_target_idx = trues_target_idx[trues_target_idx[:, 0] == i, :][:, 1]
      pred_target_idx = preds_target_idx[preds_target_idx[:, 0] == i, :][:, 1]
      true_targets = [self.config["cats_target"][j] for j in true_target_idx]
      pred_targets = [self.config["cats_target"][j] for j in pred_target_idx]

      print(texts[i])
      print(self.visualize_rationale(trues_rationale[i], spans[i]))
      print(self.visualize_rationale(preds_rationale[i], offsets[i]))
      print(true_label, pred_label)
      print(true_targets, pred_targets)

  def visualize_dataset(self):
    df = self.datamodule.df.sample(25)
    text_info = (
      df["texts"].to_list(),
      df["tokens"].to_numpy(),
      df["mask"].to_numpy(),
      df["offsets"].to_numpy(),
    )
    label_info =(
      df["label"].to_list(),
      df["target"].to_numpy(),
      df["rationales"].to_numpy(),
      df["spans"].to_list(),
    )
    self.evaluate(text_info, label_info)
    
  # works for both np padded and jagged list offsets
  def visualize_rationale(self, preds_rationale, offsets):
    out = [" "] * max(offset[1] for offset in offsets)
    for pred, offset in zip(preds_rationale, offsets):
      if pred:
        for j in range(offset[0], offset[1]):
          out[j] = "X"
    return "".join(out)

  def visualize_repl(self):
    input_text = ""
    while input_text.strip() != "q":
      input_text = input("input: ")
      self.evaluate_single(input_text)
