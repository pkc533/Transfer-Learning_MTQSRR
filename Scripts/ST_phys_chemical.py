import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
import sys

from sklearn.model_selection import KFold
import torch.nn as nn
import torch.optim as optim
import shap

import models
import preprocessing as proc
import metrics as mf


# Seed
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class SingleTargetModelling:

    def leaveOneOutValidation(dataset, learning_rate, wd, fine_tune=False, transfer_learning=False):
        DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'
        LR = learning_rate #1e-4
        WD = wd #0.5
        N_EPOCHS = 500
        FINE_TUNE_MODEL = fine_tune
        N_TARGET = 5

        # Load dataset and get features + target
        if(dataset=="LPAC"):
            X,y = proc.fetch_lpac_data()
        elif(dataset=="ACN"):
            X,y = proc.fetch_ACN_data()
        elif(dataset=="RIKEN"):
            X,y = proc.fetch_riken_data()
            N_TARGET = 1
        else:
            Exception("Dataset not valid")

        train_means = []
        val_means = []
        
        for i in range(N_TARGET):
            print(f"Target {i+1}")
            y_target = y.iloc[:, i]

            X_train_val, y_train_val, _, _ = proc.split_dataset_test(X, y_target, test_size=0.2)
            CV_FOLDS = X_train_val.shape[0]

            # Initialize your KFold object
            kf = KFold(n_splits=CV_FOLDS, shuffle=True)
            val_losses = []
            train_losses = []
            epochs = []

            for i, (train_index, val_index) in enumerate(kf.split(X_train_val)):
                print(f"****** Fold {i+1} ******")
                # Split the data into training and validation sets and scale features
                X_train, X_val = proc.scale_features(X_train_val.iloc[train_index], X_train_val.iloc[val_index])
                y_train, y_val = y_train_val.iloc[train_index], y_train_val.iloc[val_index]

                X_train, y_train = torch.tensor(X_train.values, dtype=torch.float).to(DEVICE), torch.tensor(y_train.values, dtype=torch.float).reshape(-1, 1).to(DEVICE)
                X_val, y_val = torch.tensor(X_val.values, dtype=torch.float).to(DEVICE), torch.tensor(y_val.values, dtype=torch.float).reshape(-1, 1).to(DEVICE)

                model = models.feedNN()
                if(transfer_learning):        
                    model.load_state_dict(torch.load("SMRTmodel"))
                model = model.to(DEVICE)

                for name, param in model.named_parameters():
                    if name.startswith('output'):
                        param.requires_grad = True
                    elif name.startswith('fc2'):
                        param.requires_grad = True
                    elif name.startswith('fc3'):
                        param.requires_grad = True
                    elif name.startswith('fc4'):
                        param.requires_grad = True
                    else:
                        param.requires_grad = FINE_TUNE_MODEL
                
                # Define loss and optimizer
                loss_fn = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
            
                # Train model
                for epoch in range(N_EPOCHS):
                    model.train()
                    y_pred = model(X_train)
                    loss = loss_fn(y_pred, y_train)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Evaluate your model on the validation set
                    model.eval()
                    with torch.no_grad():
                        y_pred_val = model(X_val)
                        val_loss = loss_fn(y_pred_val, y_val)
        
                train_losses.append(float(loss))
                val_losses.append(float(val_loss))
                epochs.append(float(epoch))
            
            print(f"Mean training loss over {CV_FOLDS} for best validation loss: {np.mean(train_losses)} +- {np.std(train_losses)}")
            print(f"Mean best validation loss over {CV_FOLDS}: {np.mean(val_losses)} +- {np.std(val_losses)}")
            print(f"Mean epoch at which best validation loss : {np.mean(epochs)} +- {np.std(epochs)}")
            train_means.append(np.mean(train_losses))
            val_means.append(np.mean(val_losses))


        print("--------------------------------------------------------------------------------------")
        print(f"Mean training loss for 5 targets: {np.mean(train_means)} +- {np.std(train_means)}")
        print(f"Mean validation loss  for 5 targets: {np.mean(val_means)} +- {np.std(val_means)}")

        for i in range(N_TARGET):
            print(f"Target {i+1}: {train_means[i], val_means[i]}")
    
    def test(dataset, transfer_learning):
        DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'
        LR = 1e-4
        WD = 0.01
        N_EPOCHS = 500
        N_TARGETS = 5

        if(dataset=="LPAC"):
            name_model = "LPAC"
            X, y = proc.fetch_lpac_data()
        elif(dataset=="ACN"):
            name_model = "ACN"
            X, y = proc.fetch_ACN_data()
            WD = 0.1
        elif(dataset=="RIKEN"):
            name_model = "RIKEN"
            X, y = proc.fetch_riken_data()
            N_TARGETS = 1_
        else:
            Exception("Dataset not valid")
            
        if(transfer_learning):        
            FINE_TUNE_ALL_MODEL = True if dataset != "RIKEN" else False
            name_model += " M2_phys_TL"
        else:
            name_model += " M1_phys_WTL"

        features_names = X.columns
        mse_means = []
        rmse_means = []
        mape_means = []
        r2_means = []

        y_test_all = []
        y_pred_test_all = []
        shap_values_list = []

        # Training pipeline for each target

        for i in range(N_TARGETS):
            print(f"Target {i+1}")
            y_target = y.iloc[:, i]
            X_train, y_train, X_test, y_test = proc.split_dataset_test(X, y_target, test_size=0.2)
            X_train, X_test = proc.scale_features(X_train, X_test)

            # Convert to 2D PyTorch tensors
            X_train = torch.tensor(X_train.values, dtype=torch.float32).to(DEVICE)
            y_train = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1).to(DEVICE)
            X_test = torch.tensor(X_test.values, dtype=torch.float32).to(DEVICE)
            y_test = torch.tensor(y_test.values, dtype=torch.float32).reshape(-1, 1).to(DEVICE)
            
            
            model = models.feedNN()
            if(transfer_learning):
                model.load_state_dict(torch.load("SMRTModel"))
            model = model.to(DEVICE)

            for name, param in model.named_parameters():
                if name.startswith('output'):
                    param.requires_grad = True
                elif name.startswith('fc2'):
                    param.requires_grad = True
                elif name.startswith('fc3'):
                    param.requires_grad = True
                elif name.startswith('fc4'):
                    param.requires_grad = True
                else:
                    param.requires_grad = FINE_TUNE_ALL_MODEL

            # Define loss and optimizer
            loss_fn = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

            history_train = []
            history_eval = []

            start = time.time()
            for epoch in range(N_EPOCHS):
                model.train()
                # forward pass
                y_pred = model(X_train)
                loss = loss_fn(y_pred, y_train)
                history_train.append(float(loss))
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()

                # evaluate at end of each epoch
                model.eval()
                with torch.no_grad():
                    y_pred_test = model(X_test)
                    mse = loss_fn(y_pred_test, y_test)

                mse = float(mse)
                history_eval.append(mse)
                print(f"Epoch {epoch} - Train loss {float(loss)} - Test loss {mse}")
            
            end = time.time()
            print(f"Time to train: {end - start}")
            model.eval()
            with torch.no_grad():
                y_pred_test = model(X_test)
                y_pred_test_all.append(y_pred_test.cpu())
                y_test_all.append(y_test.cpu())
                mse, rmse, mape, r2 = mf.all_metrics(y_pred_test.cpu(), y_test.cpu())
                mse_means.append(mse)
                rmse_means.append(rmse)
                mape_means.append(mape)
                r2_means.append(r2)


            # Plots
            plt.plot(history_train, label="Training loss")
            plt.plot(history_eval, label="Test loss")
            plt.title(f"Transfer learning MSE losses for fine-tuned model over epochs for target {i+1}\n")
            plt.legend()
            plt.show()

            # SHAP plot
            model.cpu()
            f = lambda x: model(torch.autograd.Variable(torch.from_numpy(x))).detach().numpy()
            explainer = shap.KernelExplainer(f, X_test.cpu().numpy())
            shap_values = explainer.shap_values(X_test.cpu().numpy())
            shap_values_list.append(shap_values)
            plt.title(f"Summary bar plot - {name_model} Target {i+1}")
            shap.summary_plot(shap_values, X_test.cpu().numpy(), features_names, plot_type="bar")
        

        print("--------------------------------------------------------------------------------------")
        print(f"Mean mse: {np.mean(mse_means)}")
        print(f"Mean rmse: {np.mean(rmse_means)}")
        print(f"Mean mape: {np.mean(mape_means)}")
        print(f"Mean r2: {np.mean(r2_means)}")

        for i in range(N_TARGETS):
            print(f"Target {i+1}: {mse_means[i], rmse_means[i], mape_means[i], r2_means[i]}")

        # Save shap values
        data_2d = np.vstack(shap_values_list)
        flattened_data = np.vstack([array.flatten() for array in data_2d])
        column_names = [f'column_{i+1}' for i in range(flattened_data.shape[1])]
        df = pd.DataFrame(flattened_data, columns=column_names)
        file_path = f"Final/shap_values/{name_model}.csv"
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        df.to_csv(file_path, index=False)
            

        # Prediction plots
        # Saving predictions vs truth values
        y_test_all_np = np.concatenate([tensor.cpu().numpy() for tensor in y_test_all], axis=1)
        y_pred_test_all_np = np.concatenate([tensor.cpu().numpy() for tensor in y_pred_test_all], axis=1)

        # Create DataFrame
        df = pd.DataFrame(np.concatenate([y_test_all_np, y_pred_test_all_np], axis=1), 
                        columns=[f'truth_target_{i+1}' for i in range(y_test_all_np.shape[1])] + 
                                [f'pred_target_{i+1}' for i in range(y_pred_test_all_np.shape[1])])
        
        file_path = f'Final/predictions/{name_model}.csv'
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        df.to_csv(file_path, index=False)

        
        fig, axes = plt.subplots(nrows=1, ncols=N_TARGETS, figsize=(15, 10), sharex=True, sharey=True)
        fig.suptitle(f"Prediction plots - {name_model}", fontsize=22)
        for i, ax in enumerate(axes.flat[:N_TARGETS]):
            y_test_flat = y_test_all[i].cpu().numpy().flatten()
            y_pred_flat = y_pred_test_all[i].cpu().numpy().flatten()
        
            m, b = np.polyfit(y_test_flat, y_pred_flat, 1)

            ax.scatter(y_test_flat, y_pred_flat, c='b')
            ax.set_title(f"Traget tR {i+1}", fontsize=20)
            ax.set_xlabel('True tR (mins)',  fontsize=18)
            ax.set_ylabel('Predicted tR (mins)',  fontsize=18)
            ax.plot(y_test_flat, y_test_flat, color='lightgray', label='1:1', linewidth=2)
            ax.plot(y_test_flat, m*y_test_flat+b, color="red", label='predicted-true', linewidth=2)
            ax.legend(fontsize=15)  # Display legend
            ax.tick_params(axis='both', which='major', labelsize=14)  # Adjust tick label size
            ax.set(adjustable='box')

        fig.tight_layout()
        plt.subplots_adjust(wspace=0.1, hspace=0.5)
        plt.show()

if __name__=="__main__":
    """
    Arguments:
    ----------

    dataset: String with the dataset to load (LPAC, ACN, RIKEN)
    operation: String to determine what to perform (validation, testing)
    fine_tune and transfer_learning: Boolean values either "yes", "true", "t" or "1" strings for setting to True
    
    To launch file:

    python ST_phys_chemical.py LPAC validation t t
    """

    dataset = sys.argv[1]
    operation = sys.argv[2]
    fine_tune = True if sys.argv[3] in ("yes", "true", "t", "1") else False
    transfer_learning = True if sys.argv[4] in ("yes", "true", "t", "1") else False

    pipeline = SingleTargetModelling()

    if(operation=="validation"):
        print("Launching valiation pipeline")
        lr_ls = [1e-4, 0.001, 0.01, 0.1]
        wd_ls = [0.5, 0.1, 0.01, 0.005, 0]
        for lr in lr_ls:
            for wd in wd_ls:
                print(f"Validation results for LR={lr} and WD={wd} and with fine-tuning for all layers f{"activated" if fine_tune else "deactivated"}")
                pipeline.leaveOneOutValidationTL(dataset, lr, wd, fine_tune, transfer_learning)
    
    elif(operation =="testing"):
        print("Launching testing pipeline")
        pipeline.test(dataset, transfer_learning) 
    else:
        Exception("Operation not available")