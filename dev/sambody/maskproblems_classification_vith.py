import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
import pickle
import random

cudnn.benchmark = True

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(518),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(518),
        transforms.CenterCrop(518),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


#data_dir = '/home/maxime/prg/phd/dev/sambody/maskproblems_classfication_split_2classes'
data_dir = '/home/maxime/prg/phd/dev/sambody/maskproblems_classification_split_2classesmoved'


image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#Check some images
def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def imshow_save(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.savefig('sts.png')
    plt.close()


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))
# Make a grid from batch
out = torchvision.utils.make_grid(inputs)
imshow_save(out, title=[class_names[x] for x in classes])


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model




def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])
                plt.pause(0.001)

                if images_so_far == num_images:
                    plt.savefig('sts.png')
                    plt.close()
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)



###ConvNet as fixed feature extractor

weights = torchvision.models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1
model_conv = torchvision.models.vit_h_14(weights=weights)


#Freeze model parameters
for param in model_conv.parameters():
    param.requires_grad = False



num_ftrs = model_conv.heads.head.in_features

model_conv.heads.head = nn.Linear(num_ftrs, len(class_names))

#model_conv = nn.DataParallel(model_conv)
model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.SGD(model_conv.heads.head.parameters(), lr=0.0001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

model_trained = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=25)


visualize_model(model_conv)

torch.save(model_trained, "classifier_perfect_pickle.pt")

model_scripted = torch.jit.script(model_trained) # Export to TorchScript
model_scripted.save('classifier_perfect_tscript.pt') # Save

###Testing
#Load with weights
weights = torchvision.models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1
modelload = torchvision.models.vit_h_14(weights=weights)

num_ftrs = modelload.heads.head.in_features
modelload.heads.head = nn.Linear(num_ftrs, 2)

modelload = modelload.to(device)
modelload.load_state_dict(torch.load('/home/maxime/prg/phd/dev/sambody/worm_noworm_classifier_vith_perfect_weights.pth', map_location=device))



#Load with torchscript
model_classif = torch.jit.load('/home/maxime/prg/phd/dev/sambody/worm_noworm_classifier_vith_perfect_tscript.pt')
model_classif.eval()

with open('/home/maxime/prg/phd/multimask_cutouts_classification.pkl', 'rb') as file:
    cutout_classification = pickle.load(file)


noclass = [image['maskid'] for image in cutout_classification if not image.get('classid')]


def visualize_model_predictions(model,img_path):
    was_training = model.training
    model.eval()

    img = Image.open(img_path)
    img = data_transforms['val'](img)
    img = img.unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)

        ax = plt.subplot(2,2,1)
        ax.axis('off')
        ax.set_title(f'Predicted: {class_names[preds[0]]}')
        imshow(img.cpu().data[0])
        plt.savefig('imgpredic.png')
        plt.close()

        model.train(mode=was_training)

visualize_model_predictions(
    #worm_noworm_classif_model,
    modelload,
    img_path='/home/maxime/prg/phd/dev/sambody/multimask_cutouts/c2_a_p3_01_d2.png'
)


visualize_model_predictions(
    modelload,
    img_path='/home/maxime/prg/phd/dev/sambody/multimask_cutouts/' + noclass[16]
)

"""
0['c1_d_p3_02_d1.png', WT (small)
1'c1_b_p1_01_d2.png', WT (dry)
2'c2_b_p1_05a_d5.png', WT (blurry nose)
3'c4_c_p3_02_d3.png', worm with bubble ----
4'c3_d_p1_03_d2.png', worm with lint ----
5'c2_c_p4_01a_d5.png', WT (curled worm with missing tail)
6'c2_b_p4_01_d1.png', NT (lint)
7'c2_d_p3_03_d1.png', worm without tail with bubble ----
8'c2_b_p2_03a_d5.png', NT (out of focus lint)
9'c2_a_p2_02_d1.png', WT (other worm cut out of frame)
10'c2_a_p4_01_d1.png', NT (out of focus lint)
11'c4_a_p3_02_d2.png', worm with bubble ----
12'c1_b_p1_03_d2.png', WT (very thin worm)
13'c1_c_p1_02_d1.png', NF (worm half out of frame and not well segmented)
14'c4_c_p2_04_d2.png', WT (blurry nose)
15'c2_d_p1_02_d2.png', WT (blurry nose)
16'c1_a_p2_01_d1.png'] WT (dry)
"""



