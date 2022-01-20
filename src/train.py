from ray import tune
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import modelnet40
from models import PointNet
from losses import OrthogonalRegularizer
from config import config, dataset, base_net
from visualize import draw_samples


class Trainable(tune.Trainable):
    def setup(self, config):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        modelnet40.create_dataset(
            dataset['src_dir'],
            dataset['n_samples'],
            dataset['n_points'],
            dataset['train_dir'],
            True
        )
        train_dataset = modelnet40.ModelNet40(dataset['train_dir'])
        self.train_loader = DataLoader(train_dataset, config['batch_size'], True)

        modelnet40.create_dataset(
            dataset['src_dir'],
            dataset['n_samples'],
            dataset['n_points'],
            dataset['test_dir'],
            True
        )
        test_dataset = modelnet40.ModelNet40(dataset['test_dir'])
        self.test_loader = DataLoader(test_dataset, config['batch_size'], True)

        self.alpha = config['alpha']

        self.net = PointNet(base_net).to(self.device)

        self.optimizer = optim.Adam(self.net.parameters(), config['lr'])

        self.ce_criterion = nn.CrossEntropyLoss()
        self.or_criterion = OrthogonalRegularizer(self.device)

        log_name = self.trial_id + '--' + \
            '--'.join(f'{key}-{value}' for key, value in config.items())
        self.writer = SummaryWriter(f'/logs/{log_name}')

        points, labels = next(iter(self.train_loader))
        sample_images = draw_samples(points, labels, test_dataset.idx_to_class, 8, 8)
        self.writer.add_image('Train Samples', sample_images, 0)

        points, labels = next(iter(self.test_loader))
        sample_images = draw_samples(points, labels, test_dataset.idx_to_class, 8, 8)
        self.writer.add_image('Test Samples', sample_images, 0)

        self.n_updates = 1

    def step(self):
        self.net.train()
        for points, labels in self.train_loader:
            self.net.zero_grad()

            points = points.to(self.device).float()
            labels = labels.to(self.device)

            preds, ws = self.net(points)

            ce_err = self.ce_criterion(preds, labels)
            re_err = 0.
            for w in ws:
                re_err = re_err + self.alpha * self.or_criterion(w)
            err = ce_err + re_err
            err.backward()

            self.optimizer.step()

            self.writer.add_scalar('Train Loss/Total', err.item(), self.n_updates)
            self.writer.add_scalar('Train Loss/CE', ce_err.item(), self.n_updates)
            self.writer.add_scalar('Train Loss/RE', re_err.item(), self.n_updates)

            acc = torch.sum(torch.argmax(preds, dim=1) == labels) / preds.size(0)

            self.writer.add_scalar('Train ACC/RE', acc.item(), self.n_updates)

            self.n_updates += 1

            # if self.n_updates % 1000 == 0:
            #     self.net.eval()
            #     val_acc = 0.
            #     n_data = 0
            #     for i, (points, labels) in enumerate(self.test_loader):
            #         points = points.to(self.device).float()
            #         labels = labels.to(self.device)
            #
            #         preds, ws = self.net(points)
            #
            #         val_acc = val_acc + torch.sum(torch.argmax(preds, dim=1) == labels)
            #         n_data = n_data + points.size(0)
            #
            #         if i >= 100:
            #             break
            #     val_acc = val_acc / n_data
            #     self.writer.add_scalar('Val Acc', val_acc.item(), self.n_updates)
            #     self.net.train()
        return {}


if __name__ == '__main__':
    tune.run(
        Trainable,
        stop={'training_iteration': 200},
        config=config,
        resources_per_trial={'gpu': 0.2, 'cpu': 0.5}
    )
