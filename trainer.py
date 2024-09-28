import torch
from dataloader import BookDataLoader
from simple_gpt import SimpleGPT
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import os
import time
from datetime import datetime, UTC


NUM_BATCHES_EVALUATE = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Trainer:

    def __init__(
        self,
        model : SimpleGPT,
        dataloader : BookDataLoader,
        lr : float,
        T_0: int,
        T_mult: int,
        save_dir : str,
    ) -> None:
        self.model = model.to(DEVICE)
        self.dataloader = dataloader
        self.save_dir = save_dir
        self.batches_trained_on = 0
        self.loss_curve = list()
        self.birthday = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M")
        self.optimizer = AdamW(self.model.parameters(), lr=lr)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=T_0, T_mult=T_mult)


    def evaluate(
        self,
        num_batches_evaluate : int,
        from_val : bool = True,
    ) -> torch.tensor:
        """Compute average loss on a handful of batches"""
        self.model.eval()
        loss_samples = list()
        for i in range(num_batches_evaluate):
            x, y = self.dataloader.get_batch(from_val=from_val)
            with torch.no_grad():
                loss = self.model(x, y)
                loss_samples.append(loss)
        self.model.train()
        return torch.tensor(loss_samples).mean().item()


    def train_on_one_batch(self):
        x, y = self.dataloader.get_batch(from_val=False)
        loss = self.model(x, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()


    def launch_train_session(
        self,
        num_batches : int,
        evaluate_every : int,
        verbose : bool = True,
    ):
        self.model.train()
        best_loss = torch.inf
        execution_time = list()
        for i in range(num_batches):
            start = time.time()
            self.train_on_one_batch()
            execution_time.append(time.time() - start)
            if i % evaluate_every == 0:
                loss = self.evaluate(NUM_BATCHES_EVALUATE)
                self.loss_curve.append((self.batches_trained_on, loss))
                if loss < best_loss:
                    self.save()
                    best_loss = loss
                if verbose:
                    print(f"Batch {self.batches_trained_on:5} | VAL LOSS = {loss:6.3f} | {torch.tensor(execution_time).mean():6.3f} s/batch | lr = {self.scheduler._last_lr[0]:.5f}")
            self.batches_trained_on += 1


    def save(self):
        torch.save(
            self.model.state_dict(), 
            os.path.join(self.save_dir, f"model_{self.birthday}.pt"),
        )

