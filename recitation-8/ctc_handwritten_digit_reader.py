import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import ctc_model
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

# seed for reproducibility
np.random.seed(785)
torch.manual_seed(785)
torch.cuda.manual_seed(785)

#################################################

# Check if data already exists and synthesize if not
if not os.path.exists("dataset") or (os.listdir("dataset")) == 0:
  import mkSeqMNIST as data_synthesizer
  data_synthesizer.make()

data, labels = np.load("dataset/data.npy"), np.load("dataset/labels.npy")

#################################################

print(data.shape)
print(labels.shape)

train = False
model_file = "example.pt" if not train else "models/checkpoint.pt"
if train:
  ctc_model.run()
model = ctc_model.DigitsModel()
model.load_state_dict(torch.load(model_file))
print(model)

#################################################

label_map = [' '] + ctc_model.DIGITS_MAP

decoder = ctc_model.CTCBeamDecoder(labels=label_map, blank_id=0)

# randomly sample 15 data points
idxs = np.random.choice(20000, 15)
data_batch, label_batch = torch.Tensor(
    data[idxs]), torch.Tensor(labels[idxs]).long()

logits, out_lengths = model(data_batch.unsqueeze(1))
label_lengths = torch.zeros((15,)).fill_(10)
logits = torch.transpose(logits, 0, 1)
probs = F.softmax(logits, dim=2).data.cpu()
output, scores, timesteps, out_seq_len = decoder.decode(
    probs=probs, seq_lens=out_lengths)
print(output.size())
print(f'out_seq_len[:, 0]: {out_seq_len[:, 0]}')

for i in range(output.size(0)):
  chrs = [label_map[o.item()] for o in output[i, 0, :out_seq_len[i, 0]]]
  image = data_batch[i].numpy()
  plt.figure()
  # imshow(image, cmap='binary')
  from scipy.misc import imsave
  imsave("img_{}.png".format("".join(chrs)), image)
  txt_top = "Prediction: {}".format("".join(chrs))
  txt_bottom = "Labelling:  {}".format(
      "".join(label_batch[i].numpy().astype(str)))
  plt.figtext(0.5, 0.10, txt_top, wrap=True,
              horizontalalignment='center', fontsize=16)
  plt.figtext(0.5, 0.01, txt_bottom, wrap=True,
              horizontalalignment='center', fontsize=16)

#################################################
