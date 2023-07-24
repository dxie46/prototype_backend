
# from sklearn.model_selection import StratifiedShuffleSplit
# from fastai.vision.all import *
# from fastai.torch_core import default_device
# from fastai.callback.tracker import EarlyStoppingCallback
# from fastai.callback.training import GradientClip

# #from timm.models import *
# import os

# from sklearn.feature_selection import RFE
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn import preprocessing
# from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
# from sklearn.model_selection import cross_val_score
# from sklearn.pipeline import Pipeline

# from sklearn.model_selection import cross_validate
# from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# from utils.parse import label_func

# import warnings

# def train():

#     warnings.filterwarnings('ignore')

#     path = "."
#     items = get_image_files('./images')

#     labels = [item.parent.name for item in items]

#     sss = StratifiedShuffleSplit(n_splits=1, test_size=0.30)

#     train_idx, valid_idx = next(sss.split(items, labels))

#     epochs=75
#     freeze_epochs=2
#     sizes = [224]
#     model_list = [mobilenet_v3_large]
#     #model_list = [efficientnetv2_xl]
#     model_dict = {}

#     for size in sizes:
#         for model_name in model_list:

#             print(model_name.__class__.__name__)
#             print(size)

#             learn = None
#             dls = None
            
#             augmentations = aug_transforms(size=size,
#                                 min_zoom=0.9,
#                                 max_zoom=1.1,
#                                 max_lighting=0.2,
#                                 max_warp=0.2,
#                                 do_flip=False,
#                                 pad_mode='zeros')

#             dls = ImageDataLoaders.from_path_func(path, 
#                                                 items, 
#                                                 label_func, 
#                                                 splitter=IndexSplitter(valid_idx),
#                                                 item_tfms=Resize(size), 
#                                                 batch_tfms=augmentations, bs=8)

#             try:
#                 learn = vision_learner(dls, model_name, normalize=True)
#             except:
#                 print(f"Pretrained weights not available for {model_name.__name__}, setting pretrained=False.")
#                 learn = vision_learner(dls, model_name, normalize=True, pretrained=False)

#             early_stop_cb = EarlyStoppingCallback(monitor='valid_loss', min_delta=0.01, patience=25)

#             # Add the GradientAccumulation callback
#             accum_steps = 8  # Set the number of accumulation steps
#             learn.add_cb(GradientAccumulation(accum_steps))

#             # Use F1Score() as a function instead of [F1Score]
#             learn.metrics = [accuracy]

#             # Train the model with the callbacks
#             learn.fine_tune(epochs, cbs=[early_stop_cb], freeze_epochs=freeze_epochs)

#             # Save the learner in the dictionary
#             model_key = f"{model_name.__name__}_{size}"
#             model_dict[model_key] = learn
#             learn.export('./transfer_learn_fastai.pkl')
