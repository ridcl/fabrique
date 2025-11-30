from dataclasses import dataclass

from tqdm import tqdm
from datasets import Dataset


@dataclass
class TrainLoopState:
    epoch: int
    step: int
    is_new_epoch: bool


def train_iterator(dataset: Dataset, max_steps=None, max_epochs=10, batch_size=1):
    step = 0
    for epoch in range(max_epochs):
        new_epoch = True
        for batch in tqdm(dataset.iter(batch_size)):
            ts = TrainLoopState(step=step, epoch=epoch, is_new_epoch=new_epoch)
            yield batch, ts
            step += 1
            new_epoch = False
            if max_steps and step >= max_steps:
                return