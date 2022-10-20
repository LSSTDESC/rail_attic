# Copyright (C) 2021,2022 Bitrateep Dey, University of Pittsburgh, USA

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import MonotonicNN
from prettytable import PrettyTable
from tqdm import trange

use_amp = True  # Flag to use automatic mixed precision


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


# Define an MLP model
class MLP(nn.Module):
    def __init__(self, input_dim=6, output_dim=1, hidden_layers=[512, 512, 512]):
        super().__init__()
        self.all_layers = [input_dim]
        self.all_layers.extend(hidden_layers)
        self.all_layers.append(output_dim)

        self.layer_list = []
        for i in range(len(self.all_layers) - 1):
            self.layer_list.append(nn.Linear(self.all_layers[i], self.all_layers[i + 1]))
            self.layer_list.append(nn.PReLU())

        self.layer_list.pop()
        # self.layer_list.append( nn.Sigmoid())
        self.layers = nn.Sequential(*self.layer_list)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.fill_(0.01)

        self.layers.apply(init_weights)

    def forward(self, x):

        return self.layers(x)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
            self, patience=7, verbose=False, delta=0, path="checkpoint.pt", trace_func=print
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class RandomDataset(Dataset):
    def __init__(self, X, Y, oversample=1):
        self.X = X
        self.Y = Y
        self.len_x = len(X)
        self.oversample = oversample

    def __len__(self):
        return int(len(self.X) * self.oversample)

    def __getitem__(self, idx):
        alpha = torch.rand(1)
        feature = torch.hstack((alpha, torch.Tensor(self.X[idx % self.len_x])))
        target = (self.Y[idx % self.len_x] <= alpha).float()
        return feature, target


def load_model(input_size, hidden_layers, checkpt_path, nn_type="monotonic", sigmoid=False, gpu_id="cuda:0"):
    # Use gpu if available
    device = torch.device(gpu_id if torch.cuda.is_available() else "cpu")
    if nn_type == "mlp":
        rhat = MLP(input_size, 1, hidden_layers).to(device)
    if nn_type == "monotonic":
        rhat = MonotonicNN(input_size, hidden_layers, nb_steps=200, dev=device, sigmoid=sigmoid).to(device)

    count_parameters(rhat)
    rhat.load_state_dict(torch.load(checkpt_path))

    return rhat


def train_local_pit(X,
                    pit_values,
                    patience=20,
                    n_epochs=1000,
                    lr=0.001,
                    weight_decay=1e-5,
                    batch_size=2048,
                    frac_mlp_train=0.9,
                    lr_decay=0.99,
                    trace_func=print,
                    oversample=1,
                    n_alpha=201,
                    checkpt_path="./checkpoint_.pt",
                    nn_type="monotonic",
                    hidden_layers=[512, 512, 512],
                    gpu_id="cuda:0",
                    sigmoid=False):
    _EPSILON = 0.01
    # Use gpu if available
    device = torch.device(gpu_id if torch.cuda.is_available() else "cpu")
    # alpha grid for validation set
    alphas_grid = np.linspace(0.001, 0.999, n_alpha)

    # Split into train and valid sets
    train_size = int(frac_mlp_train * len(X))
    valid_size = len(X) - train_size

    rnd_idx = np.random.default_rng().permutation(len(X))

    x_train_rnd = X[rnd_idx[:train_size]]
    x_val_rnd = X[rnd_idx[train_size:]]

    pit_train_rand = pit_values[rnd_idx[:train_size]]
    pit_val_rand = pit_values[rnd_idx[train_size:]]

    # Creat randomized Data set for training
    trainset = RandomDataset(x_train_rnd, pit_train_rand, oversample=oversample)

    # Create static dataset for testing
    feature_val = torch.cat(
        [
            torch.Tensor(np.repeat(alphas_grid, len(x_val_rnd)))[:, None],
            torch.Tensor(np.tile(x_val_rnd, (len(alphas_grid), 1))),
        ],
        dim=-1,
    )
    target_val = torch.Tensor(
        np.tile(pit_val_rand, len(alphas_grid))
        <= np.repeat(alphas_grid, len(x_val_rnd))
    ).float()[:, None]

    validset = TensorDataset(feature_val, target_val)

    # Create Data loader
    train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(validset, batch_size=batch_size, shuffle=False)

    # Initialize the Model and optimizer, etc.
    training_loss = []
    validation_mse = []
    validation_bce = []
    validation_weighted_mse = []
    cal_loss = []

    input_size = X.shape[1] + 1
    if nn_type == "mlp":
        rhat = MLP(input_size, 1, hidden_layers).to(gpu_id)
    if nn_type == "monotonic":
        rhat = MonotonicNN(input_size, hidden_layers, nb_steps=200, dev=device, sigmoid=sigmoid).to(device)

    count_parameters(rhat)
    # Optimizer
    optimizer = torch.optim.AdamW(rhat.parameters(), lr=lr, weight_decay=weight_decay)
    # optimizer = torch.optim.SGD(rhat.parameters(), lr=lr)
    # Use lr decay
    schedule_rule = lambda epoch: lr_decay ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedule_rule)

    # Cosine annelaing
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)

    early_stopping = EarlyStopping(
        patience=patience, verbose=True, path=checkpt_path, trace_func=trace_func
    )

    # Training loop
    for epoch in range(1, n_epochs + 1):
        training_loss_batch = []
        validation_mse_batch = []
        validation_bce_batch = []
        validation_weighted_mse_batch = []
        alpha_arr = []
        out_arr = []
        target_arr = []

        # Training
        rhat.train()  # prep model for training
        for batch, (feature, target) in enumerate(train_dataloader, start=1):
            feature = feature.to(device)  # .requires_grad_()
            target = target.to(device)  # .requires_grad_()
            alpha = feature[:, 0]
            # clear the gradients of all optimized variables
            output = rhat(feature.float())
            loss = ((output - target.float()) ** 2).sum()

            # loss = (torch.squeeze((output - target.float()) ** 2)/((_EPSILON + alpha)*(_EPSILON + 1.0 - alpha))).sum()

            # loss_fn = torch.nn.BCELoss(reduction='sum')
            # loss = loss_fn(torch.clamp(torch.squeeze(output), min=0.0, max=1.0), torch.squeeze(target))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # record training loss
            training_loss_batch.append(loss.item())
        # Validation
        rhat.eval()  # prep model for evaluation

        for feature, target in valid_dataloader:
            feature = feature.to(device)
            target = target.to(device)
            alpha = feature[:, 0]

            # forward pass: compute predicted outputs by passing inputs to the model
            output = rhat(feature.float())

            # calculate the loss
            mse = ((output - target.float()) ** 2).sum()
            # record validation loss
            validation_mse_batch.append(mse.item())

            weighted_mse = (torch.squeeze((output - target.float()) ** 2) / (
                        (_EPSILON + alpha) * (_EPSILON + 1.0 - alpha))).sum()
            validation_weighted_mse_batch.append(weighted_mse.item())

            criterion = torch.nn.BCELoss(reduction='sum')
            bce = criterion(torch.clamp(torch.squeeze(output), min=1e-6, max=0.9999999), torch.squeeze(target))
            validation_bce_batch.append(bce.item())

            alpha_arr.extend(alpha.tolist())
            out_arr.extend(torch.squeeze(output).tolist())
            target_arr.extend(torch.squeeze(target).tolist())

        out_arr = (np.array(out_arr) <= np.array(alpha_arr))
        out_arr = np.array(out_arr).reshape(-1, n_alpha)
        target_arr = np.array(target_arr).reshape(-1, n_alpha)
        cal_loss_epoch = np.mean((np.mean(out_arr, axis=0) - np.mean(target_arr)) ** 2)
        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss_epoch = np.sum(training_loss_batch) / (train_size * oversample)
        valid_mse_epoch = np.sum(validation_mse_batch) / (valid_size * n_alpha)
        valid_bce_epoch = np.sum(validation_bce_batch) / (valid_size * n_alpha)
        valid_weighted_mse_epoch = np.sum(validation_weighted_mse_batch) / (valid_size * n_alpha)
        training_loss.append(train_loss_epoch)
        validation_mse.append(valid_mse_epoch)
        validation_mse.append(valid_bce_epoch)
        validation_weighted_mse.append(valid_weighted_mse_epoch)
        cal_loss.append(cal_loss_epoch)

        epoch_len = len(str(n_epochs))

        msg = (
                f"[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] | "
                + f"train_loss: {train_loss_epoch:.5f} |\n"
                + f"valid_ece: {cal_loss_epoch:.5f} | "
                + f"valid_mse: {valid_mse_epoch:.5f} | "
                + f"valid_wght_mse: {valid_weighted_mse_epoch:.5f} | "
                + f"valid_bce: {valid_bce_epoch:.5f} | "
        )

        trace_func(msg)

        # change the lr
        scheduler.step()

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_mse_epoch, rhat)
        # early_stopping(cal_loss_epoch, rhat)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # # load the last checkpoint with the best model
    rhat.load_state_dict(torch.load(checkpt_path))
    return rhat, training_loss, (validation_mse, validation_bce)


def get_local_pit(rhat, x_test, alphas, batch_size=1, gpu_id="cuda:0"):
    # Use gpu if available

    device = torch.device(gpu_id if torch.cuda.is_available() else "cpu")
    n_test = len(x_test)
    all_betas = []
    rhat.to(device)
    n_alpha = len(alphas)
    n_batches = (n_test - 1) // batch_size + 1
    for i in trange(n_batches):
        x = x_test[i * batch_size: (i + 1) * batch_size]
        with torch.no_grad():
            all_betas_batch = rhat(
                torch.Tensor(np.hstack([np.repeat(alphas, len(x))[:, None], np.tile(x, (n_alpha, 1))])).to(device)
            ).detach().cpu().numpy().reshape(n_alpha, -1).T

        all_betas_batch[all_betas_batch < 0] = 0
        all_betas_batch[all_betas_batch > 1] = 1
        all_betas.extend(all_betas_batch)
        # print(f"Batch: {i}/{n_test}", end="\r")

    return np.array(all_betas)


def trapz_grid(y, x):
    """
    Does trapezoid integration between the same limits as the grid.
    """
    dx = np.diff(x)
    trapz_area = dx* (y[:, 1:]+y[:, :-1])/2
    integral = np.cumsum(trapz_area, axis =-1)
    return np.hstack((np.zeros(len(integral))[:,None], integral))
