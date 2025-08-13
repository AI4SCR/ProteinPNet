import os
import shutil

import torch
import torch.utils.data
# import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import argparse
import re

from helpers import makedir
import model
import push
import prune
import train_and_test as tnt
import save
from log import create_logger
from preprocess import mean, std, preprocess_input_function
import wandb
import kornia.augmentation as K
import kornia.geometry.transform as KT  # For resizing

import tifffile

parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0') # python3 main.py -gpuid=0,1,2,3
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
print(os.environ['CUDA_VISIBLE_DEVICES'])

# book keeping namings and code
from settings import base_architecture, img_size, prototype_shape, num_classes, \
                     prototype_activation_function, add_on_layers_type, ablate_prototype_selection

wandb.init()
experiment_run = wandb.run.name

base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)

model_dir = './saved_models/' + base_architecture + '/' + experiment_run + '/'
makedir(model_dir)
shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'settings.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), base_architecture_type + '_features.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'model.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'train_and_test.py'), dst=model_dir)


log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
img_dir = os.path.join(model_dir, 'img')
makedir(img_dir)
weight_matrix_filename = 'outputL_weights'
prototype_img_filename_prefix = 'prototype-img'
prototype_self_act_filename_prefix = 'prototype-self-act'
proto_bound_boxes_filename_prefix = 'bb'

# load the data
from settings import train_dir, test_dir, train_push_dir, \
                     train_batch_size, test_batch_size, train_push_batch_size, mask_prototype_distance

# normalize = transforms.Normalize(mean=mean,
#                                  std=std)

# all datasets
# train set
train_dir = "/work/FAC/FBM/DBC/mrapsoma/prometex/projects/graph_archetype_discovery/notebooks/easy_dataset_larger/train"
# train_dir = "/work/FAC/FBM/DBC/mrapsoma/prometex/projects/ProtoPNet/datasets/nsclc/train_normed_cropped"
# train_dir = "/work/FAC/FBM/DBC/mrapsoma/prometex/projects/ProtoPNet/datasets/nsclc/train_normed_cropped_only_morphology"
# train_dir = "/work/FAC/FBM/DBC/mrapsoma/prometex/projects/ProtoPNet/datasets/nsclc/train_normed_43dim"

train_push_dir = train_dir
test_dir = "/work/FAC/FBM/DBC/mrapsoma/prometex/projects/graph_archetype_discovery/notebooks/easy_dataset_larger/test"
# test_dir = "/work/FAC/FBM/DBC/mrapsoma/prometex/projects/ProtoPNet/datasets/nsclc/test_normed_cropped"
# test_dir = "/work/FAC/FBM/DBC/mrapsoma/prometex/projects/ProtoPNet/datasets/nsclc/test_normed_cropped_only_morphology"
# test_dir = "/work/FAC/FBM/DBC/mrapsoma/prometex/projects/ProtoPNet/datasets/nsclc/test_normed_43dim"


# # test_dir = "/users/lmcconn1/graph_archetype_discovery/notebooks/easy_dataset_larger/test"
train_dataset = datasets.ImageFolder(
    train_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        # transforms.RandomRotation(degrees=(-180, 180)),
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        # normalize,
    ]))
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=train_batch_size, shuffle=True,
    num_workers=1, pin_memory=False)
# push set
train_push_dataset = datasets.ImageFolder(
    train_push_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
    ]))
train_push_loader = torch.utils.data.DataLoader(
    train_push_dataset, batch_size=train_push_batch_size, shuffle=False,
    num_workers=1, pin_memory=False)
# test set
test_dataset = datasets.ImageFolder(
    test_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        # normalize,
    ]))
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=test_batch_size, shuffle=False,
    num_workers=1, pin_memory=False)

# import numpy as np

# class TiffImageFolder(datasets.ImageFolder):
#     def __init__(self, root, transform=None, target_transform=None, loader=None):
#         super().__init__(root, transform=transform, target_transform=target_transform, loader=loader or self.tiff_loader)

#     @staticmethod
#     def tiff_loader(path):
#         img = tifffile.imread(path).astype(np.float32)

#         # Convert to (C, H, W)
#         if img.ndim == 2:  # grayscale
#             img = img[None, :, :]
#         elif img.shape[-1] <= 4 or img.shape[-1] == 43:  # channels last
#             img = np.moveaxis(img, -1, 0)

#         return torch.from_numpy(img)


# class Augment43Channels(torch.nn.Module):
#     def __init__(self, img_size, is_train=True):
#         super().__init__()
#         self.img_size = img_size
#         self.is_train = is_train
#         self.aug = torch.nn.Sequential(
#             K.RandomRotation(degrees=30, p=0.5),
#             K.RandomHorizontalFlip(p=0.5),
#             K.RandomVerticalFlip(p=0.5),
#         )

#     def forward(self, x):
#         # Kornia expects batched input: (B, C, H, W)
#         is_batched = True
#         if x.ndim == 3:
#             x = x.unsqueeze(0)
#             is_batched = False

#         if self.is_train: 
#             x = self.aug(x)

#         x = KT.resize(x, (self.img_size, self.img_size), interpolation='bilinear', align_corners=False)

#         if not is_batched:
#             x = x.squeeze(0)

#         return x



# ---- Usage ----
# augment_train = Augment43Channels(img_size=img_size, is_train=True)
# augment_train_push = Augment43Channels(img_size=img_size, is_train=False)
# augment_test = Augment43Channels(img_size=img_size, is_train=False)

# train_dataset = TiffImageFolder(
#     root=train_dir,
#     transform=augment_train
# )

# train_push_dataset = TiffImageFolder(
#     root=train_push_dir,
#     transform=augment_train_push
# )

# test_dataset = TiffImageFolder(
#     root=test_dir,
#     transform=augment_test,
# )

# train_loader = torch.utils.data.DataLoader(
#     train_push_dataset, batch_size=train_batch_size, shuffle=False,
#     num_workers=1, pin_memory=False)
# train_push_loader = torch.utils.data.DataLoader(
#     train_push_dataset, batch_size=train_push_batch_size, shuffle=False,
#     num_workers=1, pin_memory=False)
# # test set

# test_loader = torch.utils.data.DataLoader(
#     test_dataset, batch_size=test_batch_size, shuffle=False,
#     num_workers=1, pin_memory=False)

# we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)
log('training set size: {0}'.format(len(train_loader.dataset)))
log('push set size: {0}'.format(len(train_push_loader.dataset)))
log('test set size: {0}'.format(len(test_loader.dataset)))
log('batch size: {0}'.format(train_batch_size))

# construct the model
ppnet = model.construct_PPNet(base_architecture=base_architecture,
                              pretrained=True, img_size=img_size,
                              prototype_shape=prototype_shape,
                              num_classes=num_classes,
                              prototype_activation_function=prototype_activation_function,
                              add_on_layers_type=add_on_layers_type,
                              ablate_prototype_selection=ablate_prototype_selection,
                            #   mask_prototype_distance=mask_prototype_distance,
)
#if prototype_activation_function == 'linear':
#    ppnet.set_last_layer_incorrect_connection(incorrect_strength=0)
ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)
class_specific = True

# define optimizer
from settings import joint_optimizer_lrs, joint_lr_step_size
joint_optimizer_specs = \
[{'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3}, # bias are now also being regularized
 {'params': ppnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
 {'params': ppnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
]
joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.1)

from settings import warm_optimizer_lrs
warm_optimizer_specs = \
[{'params': ppnet.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
 {'params': ppnet.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
]
warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

from settings import last_layer_optimizer_lr
last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

# weighting of different training losses
from settings import coefs

# number of training epochs, number of warm epochs, push start epoch, push epochs
from settings import num_train_epochs, num_warm_epochs, push_start, push_epochs, data_path

config = {
    "ablate_prototype_selection": ablate_prototype_selection,
    "base_architecture": base_architecture,
    "img_size": img_size,
    "prototype_shape": prototype_shape,
    "num_classes,": num_classes,
    "prototype_activation_function": prototype_activation_function,
    "add_on_layers_type": add_on_layers_type,
    "class_specific": class_specific,
    "mask_prototype_distance": mask_prototype_distance,

    "experiment_run": experiment_run,

    "data_path": data_path,
    "train_dir": train_dir,
    "test_dir": test_dir,
    "train_push_dir": train_push_dir,
    "train_batch_size ": train_batch_size ,
    "test_batch_size ": test_batch_size ,
    "train_push_batch_size": train_push_batch_size,
    "joint_optimizer_lrs": joint_optimizer_lrs,
    "joint_lr_step_size": joint_lr_step_size,

    "warm_optimizer_lrs": warm_optimizer_lrs,

    "last_layer_optimizer_lr": last_layer_optimizer_lr,
    "coefs": coefs,

    "num_train_epochs": num_train_epochs,
    "num_warm_epochs": num_warm_epochs,

    "push_start": push_start,
    "push_epochs": push_epochs,
}

wandb.config = config

# train the model
log('start training')
import copy
for epoch in range(num_train_epochs):
    log('epoch: \t{0}'.format(epoch))

    if ablate_prototype_selection and epoch == 20:
        print("setting requires_grad false")
        ppnet_multi.module.features.requires_grad_ = False
        ppnet_multi.module.add_on_layers.requires_grad_ = False

        ppnet_multi.module.prototype_vectors = torch.nn.Parameter(
            torch.rand(ppnet_multi.module.prototype_shape), 
            requires_grad=False
        )
        ppnet_multi.module.prototype_vectors.data = ppnet_multi.module.prototype_vectors.data.cuda()
        
    if epoch < num_warm_epochs:
        tnt.warm_only(model=ppnet_multi, log=log)
        _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer,
                      class_specific=class_specific, coefs=coefs, log=log)
    else:
        tnt.joint(model=ppnet_multi, log=log)
        joint_lr_scheduler.step()
        _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer,
                      class_specific=class_specific, coefs=coefs, log=log)

    accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                    class_specific=class_specific, log=log)
    save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'nopush', accu=accu,
                                target_accu=0.70, log=log)

    if epoch >= push_start and epoch in push_epochs: 
        push.push_prototypes(
            train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
            prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
            class_specific=class_specific,
            preprocess_input_function=None, # normalize if needed
            prototype_layer_stride=1,
            root_dir_for_saving_prototypes=img_dir, # if not None, prototypes will be saved here
            epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
            prototype_img_filename_prefix=prototype_img_filename_prefix,
            prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
            proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
            save_prototype_class_identity=True,
            log=log)
        accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                        class_specific=class_specific, log=log)
        save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'push', accu=accu,
                                    target_accu=0.70, log=log)

        if prototype_activation_function != 'linear':
            tnt.last_only(model=ppnet_multi, log=log)
            for i in range(20):
                log('iteration: \t{0}'.format(i))
                _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer,
                              class_specific=class_specific, coefs=coefs, log=log)
                accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                                class_specific=class_specific, log=log)
                save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + '_' + str(i) + 'push', accu=accu,
                                            target_accu=0.70, log=log)
   
logclose()

