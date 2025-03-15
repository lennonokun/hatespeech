import torch
import numpy as np

from transformers import AutoTokenizer

from module import HateModule

class HateVisualizer():
  def __init__(self, config):
    self.config = config
    self.tokenizer = AutoTokenizer.from_pretrained(config["model"])
    self.module = HateModule.load_from_checkpoint(
      config["best_model"],
      map_location=torch.device("cuda"),
    )
    self.module.eval()
  
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
    pred_label = self.config["labels"][pred_label_idx]
    preds_target = [self.config["targets"][idx] for idx in preds_target_idx]

    vis_rationale = self.visualize_rationale(prob_rationale > 0.5, offsets[0])

    print(f"       {vis_rationale}")
    print(f"{pred_label=}")
    print(f"{eval_prob_label=}")
    print(f"{preds_target=}")
    print(f"{eval_probs_target=}")

  def visualize_rationale(self, preds_rationale, offsets):
    out = [" "] * np.max(offsets[:, 1])
    for i, pred in enumerate(preds_rationale):
      if pred:
        for j in range(offsets[i][0], offsets[i][1]):
          out[j] = "X"
    return "".join(out)

  def visualize_repl(self):
    input_text = ""
    while input_text.strip() != "q":
      input_text = input("input: ")
      self.evaluate_single(input_text)
