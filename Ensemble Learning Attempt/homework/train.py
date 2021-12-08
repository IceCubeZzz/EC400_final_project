from planner import Planner, save_model
import torch
import torch.utils.tensorboard as tb
import numpy as np
from utils import load_data, load_dataset
import dense_transforms

# torchensemble imports
from torchensemble.utils.logging import set_logger
from torchensemble import VotingRegressor
from torchensemble.utils import io

from torch.utils.data import Dataset, DataLoader
import math


def train(args):
    planner = Planner()
    """
    Your code here, modify your HW4 code

    """
    import torch

    import inspect
    transform = eval(args.transform, {k: v for k, v in inspect.getmembers(
        dense_transforms) if inspect.isclass(v)})
    dataset = load_dataset('drive_data', transform=transform)

    dataset_len = len(dataset)
    train_length = math.ceil(dataset_len * .80)
    test_length = dataset_len - train_length

    train_data, test_data = torch.utils.data.random_split(
        dataset, [train_length, test_length])

    train_loader = DataLoader(
        train_data, num_workers=args.num_workers, batch_size=64, shuffle=True, drop_last=True)

    test_loader = DataLoader(
        test_data, num_workers=args.num_workers, batch_size=64, shuffle=True, drop_last=True)

    logger = set_logger('fussion_regressor.log')

    fusion_model = VotingRegressor(
        estimator=planner,
        n_estimators=6,
        cuda=True,
        # n_jobs=args.num_workers
    )
    criterion = torch.nn.CrossEntropyLoss()
    fusion_model.set_criterion(criterion)

    fusion_model.set_optimizer('Adam',                   # parameter optimizer
                               lr=args.learning_rate,    # learning rate of the optimizer
                               weight_decay=5e-3)        # weight decay of the optimizer

    # Set the learning rate scheduler
    fusion_model.set_scheduler(
        "CosineAnnealingLR",                    # type of learning rate scheduler
        T_max=75,
    )

    # Training
    fusion_model.fit(train_loader=train_loader,  # training data
                     test_loader=test_loader,
                     epochs=args.num_epoch,
                     save_model=True)         # the number of training epochs

    # Evaluating
    accuracy = fusion_model.predict(test_data)


def log(logger, img, label, pred, global_step):
    """
    logger: train_logger/valid_logger
    img: image tensor from data loader
    label: ground-truth aim point
    pred: predited aim point
    global_step: iteration
    """
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    fig, ax = plt.subplots(1, 1)
    ax.imshow(TF.to_pil_image(img[0].cpu()))
    WH2 = np.array([img.size(-1), img.size(-2)])/2
    ax.add_artist(plt.Circle(
        WH2*(label[0].cpu().detach().numpy()+1), 2, ec='g', fill=False, lw=1.5))
    ax.add_artist(plt.Circle(
        WH2*(pred[0].cpu().detach().numpy()+1), 2, ec='r', fill=False, lw=1.5))
    logger.add_figure('viz', fig, global_step)
    del ax, fig


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=300)
    parser.add_argument('-w', '--num_workers', type=int, default=4)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-5)
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument(
        '-t', '--transform', default='Compose([ColorJitter(0.2, 0.5, 0.5, 0.2), RandomHorizontalFlip(), ToTensor()])')

    args = parser.parse_args()
    train(args)
