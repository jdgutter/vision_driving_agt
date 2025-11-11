from .planner import Planner, save_model 
import torch
import torch.utils.tensorboard as tb
import numpy as np
from .utils import load_data
from . import dense_transforms

DEBUG_EN = True

def train(args):
    from os import path

    # Use CUDA if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if DEBUG_EN:
        print('device = ', device)

    model = Planner()
    model.to(device)
    train_logger, valid_logger = None, None
    if args.logdir is not None:
        train_logger = tb.SummaryWriter(path.join(args.logdir, 'train'))

    """
    Your code here, modify your HW4 code
    Hint: Use the log function below to debug and visualize your model
    """
    # Get model parameters
    parameters = model.parameters()
    
    # Create Loss Module
    #w = torch.as_tensor(DENSE_CLASS_DISTRIBUTION)
    #w = 1/w

    #loss_module = torch.nn.BCEWithLogitsLoss(pos_weight=w[None, :, None, None]).to(device)
    loss_module = torch.nn.MSELoss()

    # Create Optimizer
    #optimizer = torch.optim.SGD(parameters, lr=args.learning_rate, momentum=0.9)
    optimizer = torch.optim.Adam(parameters)
    
    # Create LR Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)

    # Load Data
    train_data = load_data(random_horizontal_flip=args.random_horizontal_flip, color_jitter=args.color_jitter)

    global_step = 0

    # Iterate over epochs
    for epoch in range(args.epochs):

        model.train()
        # Iterate over traning data
        for input, label in train_data:

            # Use CUDA if available to speed up training
            input = input.to(device)
            label = label.to(device)

            # Zero out gradient for this iteration
            optimizer.zero_grad()

            # Forward pass input through model to get prediction
            prediction = model(input)

            # Foward pass prediction and heatmap through loss module to get loss
            loss = loss_module(input=prediction, target=label)

            # Log loss
            train_logger.add_scalar("loss", loss, global_step=global_step)

            # Log
            if global_step % 100 == 0:
                log(train_logger, input, label, prediction, global_step)

            # Compute gradient by calling backward()
            loss.backward()

            # Update parameters for gradient descent by calling optimizer.step()
            optimizer.step()

            # Increment global step
            global_step += 1

        #model.eval()
        #for input, label in valid_data:
        #
        #    # Use CUDA if available to speed up training
        #    input = input.to(device)
        #    label = label.to(device)
        #
        #    # Forward pass input through model to get prediction
        #    validation_prediction = model(input)
        #
        #    # Log
        #    if global_step % 100 == 0:
        #        log(valid_logger, input, label, validation_prediction, global_step)

        # Log and Update Learning Rate
        train_logger.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)
        scheduler.step(loss)


    save_model(model)

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
    ax.add_artist(plt.Circle(WH2*(label[0].cpu().detach().numpy()+1), 2, ec='g', fill=False, lw=1.5))
    ax.add_artist(plt.Circle(WH2*(pred[0].cpu().detach().numpy()+1), 2, ec='r', fill=False, lw=1.5))
    logger.add_figure('viz', fig, global_step)
    del ax, fig

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--logdir')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.9, help="Learning rate of the model")
    parser.add_argument('-ep', '--epochs', type=int, default=50, help="Number of epochs to train model over")

    args = parser.parse_args()

    # Forces all Data Augmentations off
    data_augmentation_off = False

    if data_augmentation_off:
        args.random_crop = None
        args.random_horizontal_flip = False
        args.color_jitter = False
    else:
        args.random_crop = None
        args.random_horizontal_flip = True
        args.color_jitter = True

    train(args)
