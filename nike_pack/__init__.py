import datetime
import hashlib
import os
import shutil
from multiprocessing import cpu_count
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from captum.attr import IntegratedGradients
from captum.attr import LayerGradCam
from datasets import load_dataset
from dotenv import load_dotenv
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from scipy.stats import ttest_ind
from sklearn.metrics import (
    accuracy_score, roc_curve, classification_report,
    f1_score, precision_score, recall_score
)
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score
from torchvision.models import (
    efficientnet_v2_m, EfficientNet_V2_M_Weights,
    regnet_y_8gf, RegNet_Y_8GF_Weights,
    convnext_base, ConvNeXt_Base_Weights
)
from torchvision.transforms import v2 as transforms, InterpolationMode
from tqdm import tqdm

from Model.Testing import EnsembleModel
from ModelLightning.LightningModel import LightningModel
from ModelLightning.data import ToPytorchDataset
from ModelLightning.data_transforms import photo_transforms

__all__ = [
    'Image', 'torch', 'efficientnet_v2_m', 'EfficientNet_V2_M_Weights',
    'regnet_y_8gf', 'RegNet_Y_8GF_Weights', 'convnext_base', 'ConvNeXt_Base_Weights',
    'plt', 'cv2', 'make_axes_locatable', 'np', 'LayerGradCam', 'transforms',
    'InterpolationMode', 'DataLoader', 'Dataset', 'tqdm', 'load_dataset',
    'accuracy_score', 'roc_curve', 'classification_report', 'f1_score',
    'precision_score', 'recall_score', 'load_dotenv', 'os', 'Path', 'hashlib',
    'shutil', 'datetime', 'EnsembleModel', 'IntegratedGradients', 'SummaryWriter',
    'nn', 'LightningModule', 'ttest_ind', 'Accuracy', 'Precision', 'Recall', 'F1Score',
    'cpu_count', 'Trainer', 'TensorBoardLogger', 'ModelCheckpoint',
    'ToPytorchDataset', 'LightningModel', 'photo_transforms'
]


