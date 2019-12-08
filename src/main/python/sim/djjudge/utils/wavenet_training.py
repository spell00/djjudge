import torch.optim as optim
import torch.utils.data
import time
import matplotlib.pyplot as plt

from sim.djjudge.utils.model_logging import Logger
from sim.djjudge.utils.wavenet_modules import *
from torch.nn import functional as F
from sim.djjudge.utils.utils import create_missing_folders
from matplotlib import pylab
from sim.djjudge.utils.CycleAnnealScheduler import CycleScheduler


def plot_performance(loss_total, results_path, filename):
    fig2, ax21 = plt.subplots(figsize=(20, 20))
    ax21.plot(loss_total, 'b-', label='Loss:')  # plotting t, a separately
    ax21.set_xlabel('epochs')
    ax21.set_ylabel('Loss')
    handle, label = ax21.get_legend_handles_labels()
    ax21.legend(handle, label)
    fig2.tight_layout()
    # pylab.show()
    create_missing_folders(results_path + "/plots/")
    pylab.savefig(results_path + "/plots/" + filename)
    plt.close()


def print_last_loss(opt):
    print("loss: ", opt.losses[-1])


def print_last_validation_result(opt):
    print("validation loss: ", opt.validation_results[-1])


class WavenetTrainer:
    def __init__(self,
                 model,
                 dataset,
                 optimizer=optim.Adam,
                 lr=0.001,
                 weight_decay=0,
                 gradient_clipping=None,
                 logger=Logger(),
                 snapshot_path=None,
                 snapshot_name='snapshot',
                 snapshot_interval=1000,
                 dtype=torch.FloatTensor,
                 ltype=torch.LongTensor,
                 model_top=None):
        self.model = model
        self.model_top = model_top

        # Will not be updated, but maybe it should.
        # If it is, the parameters of model_top must be included in self.optimizer
        if self.model_top is not None:
            self.model_top = model_top.eval()


        self.dataset = dataset
        self.dataloader = None
        self.lr = lr
        self.weight_decay = weight_decay
        self.clip = gradient_clipping
        self.optimizer_type = optimizer
        self.optimizer = self.optimizer_type(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.logger = logger
        self.logger.trainer = self
        self.snapshot_path = snapshot_path
        self.snapshot_name = snapshot_name
        self.snapshot_interval = snapshot_interval
        self.dtype = dtype
        self.ltype = ltype

    def train(self,
              batch_size=32,
              epochs=10,
              continue_training_at_step=0):
        self.model.train()
        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                      batch_size=batch_size,
                                                      shuffle=True,
                                                      num_workers=0,
                                                      pin_memory=False)
        self.scheduler = CycleScheduler(self.optimizer, self.lr, n_iter=epochs * len(self.dataloader))

        step = continue_training_at_step
        losses = []
        x_cond = None
        for current_epoch in range(epochs):
            print("epoch", current_epoch)
            tic = time.time()
            for i, (x, target) in enumerate(self.dataloader):
                print(i, '/', len(self.dataloader))
                x = x.type(self.dtype).clone().detach().requires_grad_(True)
                target = target.view(-1).type(self.ltype)
                if self.model_top is not None:
                    x_cond = self.model_top(x).reshape(x.shape[0], x.shape[1], -1)
                    # x = torch.cat([x, x_cond], 2)
                output = self.model(x, cond=x_cond)
                loss = F.cross_entropy(output.squeeze(), target.squeeze())
                self.optimizer.zero_grad()
                loss.backward()
                loss = loss.item()
                losses += [loss]
                if self.clip is not None:
                    torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clip)
                self.optimizer.step()
                step += 1

                # time step duration:
                if step == 100:
                    toc = time.time()
                    print("one training step does take approximately " + str((toc - tic) * 0.01) + " seconds)")
                if step % 10 == 0:
                    plot_performance(loss_total=losses, results_path="figures", filename="training_loss_trace_wavenet")

                if step % self.snapshot_interval == 0:
                    if self.snapshot_path is None:
                        continue
                    time_string = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
                    torch.save(self.model, self.snapshot_path + '/' + self.snapshot_name + '_' + time_string)

                self.logger.log(step, loss)

    def validate(self):
        self.model.eval()
        self.dataset.train = False
        total_loss = 0
        accurate_classifications = 0
        for (x, target) in iter(self.dataloader):
            x = x.type(self.dtype).clone().detach().requires_grad_(True)
            target = target.view(-1).type(self.ltype)

            output = self.model(x)
            loss = F.cross_entropy(output.squeeze(), target.squeeze())
            total_loss += loss.item()

            predictions = torch.max(output, 1)[1].view(-1)
            correct_pred = torch.eq(target, predictions)
            accurate_classifications += torch.sum(correct_pred).item()
        # print("validate model with " + str(len(self.dataloader.dataset)) + " samples")
        # print("average loss: ", total_loss / len(self.dataloader))
        avg_loss = total_loss / len(self.dataloader)
        avg_accuracy = accurate_classifications / (len(self.dataset) * self.dataset.target_length)
        self.dataset.train = True
        self.model.train()
        return avg_loss, avg_accuracy


def generate_audio(model,
                   length=8000,
                   temperatures=[0., 1.]):
    samples = []
    for temp in temperatures:
        samples.append(model.generate_fast(length, temperature=temp))
    samples = np.stack(samples, axis=0)
    return samples
