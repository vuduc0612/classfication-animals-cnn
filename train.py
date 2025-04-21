import numpy as np
import torch
import torch.nn as nn
from sentry_sdk.utils import epoch
from torchvision.models import resnet18, ResNet18_Weights
from torchinfo import summary
from models import CNN
from datasets import AnimalDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Resize, Compose, Normalize
from tqdm.autonotebook import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
import os
import matplotlib.pyplot as plt
import numpy as np
import warnings
import argparse
warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser(description="Animal classifier")
    parser.add_argument("--data-path", "-d", type=str, default="data", help="path to dataset")
    parser.add_argument("--log-path", "-o", type=str, default="my_tensorboard", help="tensorboard folder")
    parser.add_argument("--checkpoint-path", "-c", type=str, default="my_models", help="checkpoint folder")
    parser.add_argument("--resume-training", "-r", type=bool, default=False, help="Continue training or not")
    parser.add_argument("--image-size", "-i", type=int, default=224, help="common size of all images")
    parser.add_argument("--batch-size", "-b", type=int, default=64, help="batch size of training procedure")
    parser.add_argument("--epochs","-e", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", "-l",type=float, default=1e-3, help="learning rate of optimizer")
    parser.add_argument("--momentum", "-m",type=float, default=0.9, help="momentum of optimizer")

    args = parser.parse_args()
    return args

def plot_confusion_matrix(writer, cm, class_names, epoch):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(20, 20))
    # color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    plt.imshow(cm, interpolation='nearest', cmap="hsv")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)

def train(args):
    transform = Compose([
        ToTensor(),
        Resize((args.image_size, args.image_size)),
        Normalize([0.485, 0.546, 0.406], [0.229, 0.224, 0.225])
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = AnimalDataset(root=args.data_path, train=True, transform=transform)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    val_dataset = AnimalDataset(root=args.data_path, train=False, transform=transform)
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )
    # model = CNN(num_classes=len(train_dataset.categories))
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    del model.fc
    model.fc = nn.Linear(in_features=in_features, out_features=len(train_dataset.categories), bias=True)

    # summary(model, input_size=(1, 3, args.image_size, args.image_size))
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    if args.resume_training:
        checkpoint = os.path.join(args.checkpoint_path, "last.pt")
        saved_data = torch.load(checkpoint)
        model.load_state_dict(saved_data["model"])
        optimizer.load_state_dict(saved_data["optimizer"])
        start_epoch = saved_data["epoch"]
        best_acc = saved_data["best_acc"]
    else:
        start_epoch = 0
        best_acc = -1

    num_iters = len(train_dataloader)
    if not os.path.isdir(args.log_path):
        os.makedirs(args.log_path)
    writer = SummaryWriter(args.log_path)
    if not os.path.isdir(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    for epoch in range(start_epoch, args.epochs):
        # TRAIN
        model.train()
        progress_bar = tqdm(train_dataloader, colour="cyan")
        total_losses = []
        for iter, (images, labels) in enumerate(progress_bar):
            # forward pass
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            loss = criterion(output, labels)
            total_losses.append(loss.item())
            avg_loss = np.mean(total_losses)
            progress_bar.set_description("Epoch {}/{}. Loss {:0.4f}".format(epoch+1, args.epochs, avg_loss))
            writer.add_scalar("Train/Loss", avg_loss, global_step=epoch*num_iters+iter)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # VALIDATION
        model.eval()
        total_losses = []
        all_labels = []
        all_predictions = []
        progress_bar = tqdm(val_dataloader, colour="yellow")
        # with torch.inference_mode():    # From pytorch 1.9
        with torch.no_grad():
            for iter, (images, labels) in enumerate(progress_bar):
                # forward pass
                images = images.to(device)
                all_labels.extend(labels)
                labels = labels.to(device)
                output = model(images)   # shape [batch_size, num_classes]

                prediction = torch.argmax(output, dim=1)
                all_predictions.extend(prediction.tolist())

                loss = criterion(output, labels)
                total_losses.append(loss.item())

        avg_loss = np.mean(total_losses)
        writer.add_scalar("Val/Loss", avg_loss, global_step=epoch)
        accuracy = accuracy_score(all_labels, all_predictions)
        writer.add_scalar("Val/Accuracy", accuracy, global_step=epoch)
        plot_confusion_matrix(writer, confusion_matrix(all_labels, all_predictions), train_dataset.categories, epoch)
        print("Epoch {}. Average loss {:0.4f}. Accuracy {:0.4f}".format(epoch+1, avg_loss, accuracy))

        saved_data = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch+1,
            "best_acc": best_acc,
        }
        checkpoint = os.path.join(args.checkpoint_path, "last.pt")
        torch.save(saved_data, checkpoint)
        if accuracy > best_acc:
            checkpoint = os.path.join(args.checkpoint_path, "best.pt")
            torch.save(saved_data, checkpoint)
            best_acc = accuracy



if __name__ == '__main__':
    args = get_args()
    train(args)


