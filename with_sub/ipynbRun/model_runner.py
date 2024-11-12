
OUTPUT_FOLDER='./output/'

import os
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import torch
import random
import numpy as np
import mne
import torch.nn.functional as F

import torch
import numpy as np
import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import confusion_matrix
import os
import time
import importlib.util
import importlib.util
import os
import time
import torch
import argparse


"""### args"""

import json



def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def add_noise(data):
    if np.random.rand() > 0.5:
        data += np.random.normal(0, 0.01, (data.shape))
    return data

def handle_mixup(idx, data, label, mixup_data, supervised_mixup_data, mixup_idx, mixup_rate):
    mixup_data = mixup_data
    # use supervised mixup data if available
    if supervised_mixup_data is not None:
        mixup_data = supervised_mixup_data[label]
    # if mixup condition is set and id is not mixable
    if mixup_idx is not None and not mixup_idx[idx]:
        mixup_data = None
    # do mixup if available
    if mixup_data is not None:
        data = mixup(data, mixup_data, mixup_rate)

    return data

def mixup(data, mixup_data, mixup_rate):
    mixup_rate = np.random.rand() * mixup_rate
    idx = np.random.randint(0, len(mixup_data))
    return (1-mixup_rate)*data + mixup_rate*mixup_data[idx]

class SleepDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, transform=None, mixup_data=None, supervised_mixup_data=None, mixup_idx=None, mixup_rate=None):
        self.data = data
        self.labels = labels
        self.transform = transform

        self.supervised_mixup_data = supervised_mixup_data
        self.mixup_data = mixup_data
        self.mixup_idx = mixup_idx
        self.mixup_rate = mixup_rate

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]

        if self.labels is not None:
            label = self.labels[idx].astype(np.int64)
        else:
            label = -1

        ### DATA AUGMENTATION ###
        data = handle_mixup(idx, data, label, self.mixup_data, self.supervised_mixup_data, self.mixup_idx, self.mixup_rate)

        if self.transform:
            data = self.transform(data)
        ### END DATA AUGMENTATION ###

        data = data.astype(np.float32)

        return {"eeg": data, "label": label}


class FocalLoss(torch.nn.Module):
    def __init__(
        self,
        alpha: int = 1,
        gamma: int = 2,
        logits: bool = True,
        reduce: bool = True,
        ls: float = 0.05,
        classes: int = 6,
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        self.ls = ls
        self.classes = classes

    def forward(self, inputs, targets):
        targets = F.one_hot(targets, num_classes=6)

        if self.ls is not None:
            targets = (1 - self.ls) * targets + self.ls / self.classes

        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction="none")

        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):

    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


class VATLoss(nn.Module):

    def __init__(self, xi=10.0, eps=1.0, ip=1):
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x):
        with torch.no_grad():
            pred = F.softmax(model(x), dim=1)

        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                pred_hat = model(x + self.xi * d)
                logp_hat = F.log_softmax(pred_hat, dim=1)
                adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()

            # calc LDS
            r_adv = d * self.eps
            pred_hat = model(x + r_adv)
            logp_hat = F.log_softmax(pred_hat, dim=1)
            lds = F.kl_div(logp_hat, pred, reduction='batchmean')

        return lds

def train_epoch(args, model, loader, criterion, optimizer, scheduler, epoch):
    losses = []
    targets_all = []
    outputs_all = []

    vat_loss = VATLoss(xi=10.0, eps=1.0, ip=1)

    model.train()
    t = tqdm(loader)

    for i, sample in enumerate(t):
        optimizer.zero_grad()

        eeg = sample["eeg"].to(args.device)
        target = sample["label"].to(args.device)

        lds = vat_loss(model, eeg)
        output = model(eeg)
        loss = criterion(output, target) + args.alpha * lds
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        losses.append(loss.item())

        target = target.cpu().numpy()
        output = output.detach().cpu().numpy()

        targets_all.extend(target)
        outputs_all.extend(output)

        output_loss = np.mean(losses)
        output_score = np.mean(targets_all == np.argmax(outputs_all, axis=1))

        t.set_description(
            f"Epoch {epoch}/{args.epochs} - Train loss: {output_loss:0.4f}, score: {output_score:0.4f}"
        )

    return targets_all, outputs_all, output_score, output_loss


def validate(args, model, loader, criterion, desc="Valid"):
    losses = []
    targets_all = []
    outputs_all = []

    t = tqdm(loader)
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(t):
            eeg = sample["eeg"].to(args.device)
            target = sample["label"].to(args.device)

            output = model(eeg)
            loss = criterion(output, target)

            losses.append(loss.item())
            targets_all.extend(target.cpu().numpy())
            outputs_all.extend(output.detach().cpu().numpy())

            output_loss = np.mean(losses)
            output_score = np.mean(targets_all == np.argmax(outputs_all, axis=1))

            t.set_description(
                f"\t  - {desc} loss: {output_loss:0.4f}, score: {output_score:0.4f}"
            )

    return targets_all, outputs_all, output_score, output_loss


def predict(args, model, loader):
    outputs_all = []

    t = tqdm(loader)
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(t):
            eeg = sample["eeg"].to(args.device)

            output = model(eeg)
            outputs_all.extend(output.detach().cpu().numpy())
    return outputs_all



class base_args(object):
    def __init__(self, model_name, seed=42):
        self.model_name = model_name
        self.seed = seed

        self.lr = 1e-3
        self.epochs = 5
        self.batch_size = 120
        self.num_workers = 2
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
        self.alpha = 0.01
        self.phase = "base"

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def get_model_name(self):
        return f"{self.model_name}_{self.phase}"

"""### experiment helper functions"""

def train_model(args, model,
          train_data, train_labels,
          valid_data, valid_labels,
          train_weights=None, sample_rate=None,
          use_scheduler=True,
          history = {"Train": {"Score": [], "Loss": []},
                     "Valid": {"Score": [], "Loss": []}},
          mixup_data=None, supervised_mixup_data=None, mixup_idx=None, mixup_rate=None,filename=None, log_to_file=False):

    train_dataset = SleepDataset(train_data, train_labels, transform=add_noise,
                                 mixup_data=mixup_data, supervised_mixup_data=supervised_mixup_data,
                                 mixup_idx=mixup_idx, mixup_rate=mixup_rate)

    if train_weights is not None:
        train_sampler = WeightedRandomSampler(weights = train_weights, num_samples=int(len(train_labels)*sample_rate))
        train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, sampler=train_sampler, drop_last=True)
    else:
        train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = model.to(args.device)
    criterion = FocalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)

    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
                        optimizer,
                        max_lr=args.lr,
                        epochs=args.epochs,
                        steps_per_epoch=len(train_loader),
                        div_factor=10,
                        final_div_factor=10,
                        pct_start=0.1,
                        anneal_strategy="cos",
                    )
    else:
        scheduler = None

    for epoch in range(1, args.epochs+1):
        _, _, train_score, train_loss = train_epoch(args, model, train_loader, criterion, optimizer, scheduler, epoch)
        _, _, valid_score, valid_loss = validate_model(args, model, valid_data, valid_labels, show_plot=False)

        history["Train"]["Loss"].append(train_loss)
        history["Train"]["Score"].append(train_score)
        history["Valid"]["Loss"].append(valid_loss)
        history["Valid"]["Score"].append(valid_score)
	
    torch.save(model.state_dict(), f"{OUTPUT_FOLDER}best_model_{args.get_model_name()}.pt")
    if log_to_file:
        print("train loss: " + str(history["Train"]["Loss"][-1]))
        print("train score: " + str(history["Train"]["Score"][-1]))
        print("val loss: " + str(history["Valid"]["Loss"][-1]))
        print("val score: " + str(history["Valid"]["Score"][-1]))

        if filename is not None:
            with open(filename, "a") as f:
                f.write("\n train loss: " + str(history["Train"]["Loss"][-1]))
                f.write("\n train score: " + str(history["Train"]["Score"][-1]))
                f.write("\n val loss: " + str(history["Valid"]["Loss"][-1]))
                f.write("\n val score: " + str(history["Valid"]["Score"][-1]))
	



def validate_model(args, model, valid_data, valid_labels, desc="Target", show_plot=True):
    criterion = FocalLoss()

    valid_dataset = SleepDataset(valid_data, valid_labels)
    valid_loader = DataLoader(valid_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)
    targets_all, outputs_all, output_score, output_loss = validate(args, model, valid_loader, criterion, desc)
	
    if show_plot:
        cf_mat = confusion_matrix(targets_all, np.argmax(outputs_all, axis=1), normalize="true")
        plt.figure()
        sns.heatmap(cf_mat, annot=True)
        plt.show()

    return targets_all, outputs_all, output_score, output_loss


def get_prediction(args, model, data):
    model.to(args.device)
    model.load_state_dict(torch.load(f"{OUTPUT_FOLDER}best_model_{args.get_model_name()}.pt"))

    dataset = SleepDataset(data, None)
    loader = DataLoader(dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    output = predict(args, model, loader)
    return output


def ensemble(args_array, dataset_name="test", phase=None):
    ensemble_output = None

    for args in args_array:
        if phase is None:
            fn = f"{OUTPUT_FOLDER}{args.get_model_name()}-{dataset_name}_output.npy"
        else:
            fn = f"{OUTPUT_FOLDER}{args.model_name}_{phase}-{dataset_name}_output.npy"

        output = np.load(fn)
        if ensemble_output is None:
            ensemble_output = output
        else:
            ensemble_output += output

    return ensemble_output


def predict_and_save(args, model, target_data, test_data):
    target_output = get_prediction(args, model, target_data)
    np.save(f"{OUTPUT_FOLDER}{args.get_model_name()}-target_output.npy", target_output)

    test_output = get_prediction(args, model, test_data)
    np.save(f"{OUTPUT_FOLDER}{args.get_model_name()}-test_output.npy", test_output)


def plot_history(history):
    fig, axes = plt.subplots(2,1, figsize=(22,6))
    axes[0].plot(history["Train"]["Score"], label="Train score")
    axes[0].plot(history["Valid"]["Score"], label="Valid score")
    axes[0].legend()
    axes[1].plot(history["Train"]["Loss"], label="Train loss")
    axes[1].plot(history["Valid"]["Loss"], label="Valid loss")
    axes[1].legend()
    fig.show()

"""### competition data loaders

### training pipeline function
"""

def supervised_run(args, model, train_data, train_labels, target_data, target_labels, test_data, mixup_data=None, supervised_mixup_data=None, filename = None):
    print(args.toJSON())

    history = {"Train": {"Score": [], "Loss": []},
               "Valid": {"Score": [], "Loss": []}}

    ############################
    ### USE ONLY SOURCE DATA ###
    ############################
    args.phase = "base"

    train_model(args, model,
                train_data, train_labels,
                target_data, target_labels,
                mixup_data=mixup_data, mixup_rate=0.,
                history=history)
    # validate
    model.load_state_dict(torch.load(f"{OUTPUT_FOLDER}best_model_{args.get_model_name()}.pt"))
    print("\n###### PHASE 1 FINISHED ##########")
    validate_model(args, model, target_data, target_labels, "Target")
    # make predictions and save results
    predict_and_save(args, model, target_data, test_data)


    ##############################
    ### SOURCE DATA WITH SUPERVISED MIXUP ###
    ##############################
    args.lr = 1e-4
    args.epochs = 10
    args.phase = "mixup"

    train_model(args, model,
                train_data, train_labels,
                target_data, target_labels,
                supervised_mixup_data=supervised_mixup_data, mixup_rate=0.5,
                history=history)
    # validate
    model.load_state_dict(torch.load(f"{OUTPUT_FOLDER}best_model_{args.get_model_name()}.pt"))
    print("\n###### PHASE 2 FINISHED ########OUTPUT_FOLDER##")
    validate_model(args, model, target_data, target_labels, "Target")
    # make predictions and save results
    predict_and_save(args, model, target_data, test_data)

    ###############################
    ### FINETUNE ON TARGET DATA WITH MIXED UP SOURCE ###
    ###############################
    args.lr = 1e-3
    args.epochs = 20
    args.phase = "mixup_finetuned"
    # prepare extended train data with sampler settings
    extended_train_data = np.concatenate((train_data, target_data), axis=0)
    extended_train_labels = np.concatenate((train_labels, target_labels), axis=0)

    train_weights = [0.25] * len(train_labels) + [0.75] * len(target_labels)
    mixup_idx = [True] * len(train_labels) + [False] * len(target_labels)
    train_sample_rate = 0.5

    train_model(args, model,
                extended_train_data, extended_train_labels,
                target_data, target_labels,
                train_weights = train_weights, sample_rate=train_sample_rate,
                mixup_data=mixup_data, mixup_idx=mixup_idx, mixup_rate=0.5,
                use_scheduler=True, history=history,filename =filename, log_to_file=True)

    # validate
    model.load_state_dict(torch.load(f"{OUTPUT_FOLDER}best_model_{args.get_model_name()}.pt"))
    print("\n###### PHASE 3 FINISHED ##########")
    validate_model(args, model, target_data, target_labels, "Target")
    # make predictions and save results
    predict_and_save(args, model, target_data, test_data)
    # plot training history
    plot_history(history)
    

def parse_args():
    parser = argparse.ArgumentParser(description="Run model training and testing.")
    parser.add_argument('model_file', type=str, help='Path to the model file (e.g., eeg_model.py)')
    args = parser.parse_args()
    return args.model_file

def modify_output_folder(model_file):
    global OUTPUT_FOLDER
    OUTPUT_FOLDER = OUTPUT_FOLDER + f"{model_file}/"
    
# Function to dynamically load the model from the given file
def load_model_from_file(model_file, model_folder=''):
    model_name = os.path.splitext(os.path.basename(model_file))[0]  # Extract model name without extension
    model_path = os.path.join(model_folder, model_file)
    
    # Load the model file dynamically
    spec = importlib.util.spec_from_file_location(model_name, model_path)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)

    # Assuming the model file has a class `EEGClassifier`
    model_class = getattr(model_module, 'EEGClassifier')
    return model_class(), model_name  # Return model instance and model name

# Function to run the model training and testing, and save the metadata
def run_and_save_model(args, model_name, source_data, source_labels, target_data, target_labels, test_data,test_labels, mixup_data=None, supervised_mixup_data=None):
    # Load the model dynamically from the specified file
    model, model_base_name = load_model_from_file(model_name)

    # Create a folder to save the model and training details
    model_dir = os.path.join(OUTPUT_FOLDER, "models", model_base_name)
    os.makedirs(model_dir, exist_ok=True)

    # Define the output metadata file
    meta_file_path = os.path.join(model_dir, f"{model_base_name}_output_meta.txt")

    # Track start time for training
    start_time = time.time()

    # Call the existing supervised_run function (no modifications)
    supervised_run(args, model, source_data, source_labels, target_data, target_labels, test_data, mixup_data=mixup_data, supervised_mixup_data=supervised_mixup_data, filename =  meta_file_path)

    # Calculate and save training time
    end_time = time.time()
    train_time = end_time - start_time
    train_time_str = time.strftime("%Hh %Mm %Ss", time.gmtime(train_time))
    print(f"Training completed in: {train_time_str}")

    # Save the training time and details to the metadata file
    with open(meta_file_path, "w") as f:
        f.write(f"Training time: {train_time_str}\n")

    # Save the trained model
    torch.save(model.state_dict(), os.path.join(model_dir, f"{model_base_name}_final_model.pt"))

    # Track start time for testing/validation
    start_test_time = time.time()

    # Validate and save predictions (already done in supervised_run)
    # Capture validation and test accuracy
    valid_score = validate_and_test(args, model, target_data, target_labels, test_data,test_labels)

    # Calculate and save testing time
    end_test_time = time.time()
    test_time = end_test_time - start_test_time
    test_time_str = time.strftime("%Hh %Mm %Ss", time.gmtime(test_time))
    # print(f"Testing completed in: {test_time_str}")

    # Save the testing time, accuracies, and losses to the metadata file
    with open(meta_file_path, "a") as f:
        # f.write(f"Testing time: {test_time_str}\n")
        f.write(f"Validation Accuracy: {valid_score:.4f}\n")
        # f.write(f"Test Accuracy: {test_score:.2f}\n")

    # Print accuracies explicitly
    print(f"Validation Accuracy: {valid_score:.4f}")
    # print(f"Test Accuracy: {test_score:.2f}")
    start_test_time = time.time()
    _, _, test_score, test_loss = validate_model(args, model, test_data, test_labels, show_plot=False)
    end_test_time = time.time()
    test_time = end_test_time - start_test_time
    test_time_str = time.strftime("%Hh %Mm %Ss", time.gmtime(test_time))
    with open(meta_file_path, "a") as f:
        f.write(f"Testing time: {test_time_str}\n")
        f.write(f"Test Accuracy: {test_score:.4f}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
		
	
	

	


from sklearn.metrics import accuracy_score
import numpy as np

def calculate_test_accuracy(predictions, true_labels):
    """
    Calculate accuracy by comparing predictions to the true labels using scikit-learn.
    Arguments:
    - predictions: List or numpy array of model predictions (logits or probabilities)
    - true_labels: Numpy array of actual class labels

    Returns:
    - accuracy: Calculated accuracy as a percentage
    """
    # Ensure predictions are a NumPy array
    if isinstance(predictions, list):
        predictions = np.array(predictions)
    
    # Ensure true_labels are a NumPy array
    if not isinstance(true_labels, np.ndarray):
        true_labels = np.array(true_labels)

    # If predictions are continuous (logits/probabilities), convert to class labels using argmax
    if predictions.ndim > 1:
        predicted_classes = np.argmax(predictions, axis=1)
    else:
        predicted_classes = predictions

    # Ensure both true_labels and predicted_classes are the same type
    predicted_classes = predicted_classes.astype(int)
    true_labels = true_labels.astype(int)

    # Calculate accuracy using scikit-learn's accuracy_score
    accuracy = accuracy_score(true_labels, predicted_classes) * 100  # Convert to percentage
    return accuracy



def validate_and_test(args, model, target_data, target_labels, test_data,test_labels):
    """
    Function to handle validation and test, returning their accuracies.
    """
    # Validate
    _, _, valid_score, _ = validate_model(args, model, target_data, target_labels, "Target")

    # Test (Using predict_and_save if it also returns test accuracy)
    test_output = get_prediction(args, model, test_data)
    # print("---------------------------------------------------------")
    # print(type(test_labels))
    # print(type(test_output))
    # print("----------------------------------------------------------")
    # test_accuracy = calculate_test_accuracy(test_labels, test_output)

    return valid_score



"""### training"""

import numpy as np

# Load train data and labels
source_data = np.load('./data/train/data.npy')
source_labels = np.load('./data/train/labels.npy')

# Load validate data and labels
target_data = np.load('./data/validate/data.npy')
target_labels = np.load('./data/validate/labels.npy')

# Load test data and labels
test_data = np.load('./data/test/data.npy')
test_labels = np.load('./data/test/labels.npy')


import numpy as np

# Assuming y_train is your array
unique_classes, counts = np.unique(target_labels, return_counts=True)

# Calculating the total number of entries
total_entries = counts.sum()

# Creating a dictionary to show class: percentage of entries
class_percentages = {cls: (count / total_entries) * 100 for cls, count in zip(unique_classes, counts)}
print(class_percentages)

# mixup_data

tmp = np.array(target_data)
supervised_mixup_data = {}
for c in np.unique(target_labels):
    supervised_mixup_data[c] = tmp[target_labels == c]
    print(c, np.shape(supervised_mixup_data[c]))

del tmp


# Print the shape of each
print("Train data shape:", source_data.shape)
print("Train labels shape:", source_labels.shape)
print("Validate data shape:", target_data.shape)
print("Validate labels shape:", target_labels.shape)
print("Test data shape:", test_data.shape)
print("Test labels shape:", test_labels.shape)

# Calculate and print total data size (total number of samples)
total_data_size = source_labels.shape[0] + target_data.shape[0] + test_data.shape[0]
print("Total data size (number of samples):", total_data_size)



# Main runner function
if __name__ == "__main__":
    # Parse model file argument from command-line
    model_file = parse_args()
    # model_file = "eeg_model.py"
    # modify_output_folder(model_file)
    # Example usage
    args_array = []
    args = base_args("eeg-classifier_seed-42")
    seed_everything(args.seed)

    # Call the new function that tracks time and saves the model using the model file name from the argument
    run_and_save_model(args, model_file, source_data, source_labels, target_data, target_labels, test_data,test_labels, mixup_data=None, supervised_mixup_data=supervised_mixup_data)

    # Append the arguments to the array for later reference
    args_array.append(args)


