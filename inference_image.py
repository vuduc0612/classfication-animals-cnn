import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import os
import cv2
import warnings
import argparse
warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser(description="Animal classifier")
    parser.add_argument("--image-path", "-p", type=str, default="test_images/3.jpg", help="Path to test image")
    parser.add_argument("--checkpoint-path", "-c", type=str, default="my_models", help="checkpoint folder")
    parser.add_argument("--image-size", "-i", type=int, default=224, help="common size of all images")

    args = parser.parse_args()
    return args


def inference(args):
    categories = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider",
                  "squirrel"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = CNN(num_classes=len(train_dataset.categories))
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    del model.fc
    model.fc = nn.Linear(in_features=in_features, out_features=len(categories), bias=True)
    model.to(device)
    checkpoint = os.path.join(args.checkpoint_path, "best.pt")
    saved_data = torch.load(checkpoint)
    model.load_state_dict(saved_data["model"])
    model.eval()

    mean = np.array([0.485, 0.546, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    ori_image = cv2.imread(args.image_path)
    image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (args.image_size, args.image_size))
    image = image/255
    image = (image - mean) / std
    image = np.transpose(image, (2, 0, 1))
    # image = np.expand_dims(image, axis=0)
    image = image[None, :, :, :]
    image = torch.from_numpy(image).float().to(device)
    softmax = nn.Softmax()
    with torch.no_grad():
        output = model(image)[0]
        predicted_class = categories[torch.argmax(output)]
        prob = softmax(output)[torch.argmax(output)]
    cv2.imshow("{} ({:0.2f}%)".format(predicted_class, prob*100), ori_image)
    cv2.waitKey(0)






if __name__ == '__main__':
    args = get_args()
    inference(args)


