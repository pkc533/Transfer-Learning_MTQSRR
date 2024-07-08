import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error

import time
import copy
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import random
import sys

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

class SmrtModelling:

    def validation(learning_rate, wd, scale_all=False,):
        DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'
        LEARNING_RATE=learning_rate
        WD = wd
        N_EPOCHS = 500

        # Load dataset and get features + target
        X, y = proc.fetch_SMRT()
        X_train, y_train, X_val, y_val, _, _ = proc.split_dataset_val(X, y)

        if(scale_all):
            X_train, y_train, X_val, y_val, scaler = proc.scale_all(X_train, y_train, X_val, y_val)
        else:
            X_train, X_val = proc.scale_features(X_train, X_val)
        
        # Convert to 2D PyTorch tensors
        X_train = torch.tensor(X_train.values, dtype=torch.float32).to(DEVICE)
        y_train = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1).to(DEVICE)
        X_val = torch.tensor(X_val.values, dtype=torch.float32).to(DEVICE)
        y_val = torch.tensor(y_val.values, dtype=torch.float32).reshape(-1, 1).to(DEVICE)

        model = models.feedNN().to(DEVICE)

        # Define loss and optimizer
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WD)

        # Hold the best model
        best_mse = np.inf   # init to infinity
        best_epoch = 0
        best_weights = None
        history_train = []
        history_eval = []
        history_eval_unscaled = []

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

            # evaluate accuracy at end of each epoch
            model.eval()
            with torch.no_grad():
                y_pred = model(X_val)
                mse = loss_fn(y_pred, y_val)
                if(scale_all):
                    y_pred_unscaled = scaler.inverse_transform(y_pred.cpu())
                    y_val_unscaled = scaler.inverse_transform(y_val.cpu())
                    y_pred_unscaled = torch.from_numpy(y_pred_unscaled).to(DEVICE)
                    y_val_unscaled = torch.from_numpy(y_val_unscaled).to(DEVICE)
                    mse_unscaled = loss_fn(y_pred_unscaled, y_val_unscaled)
                    history_eval_unscaled.append(float(mse_unscaled))
                    print(f"Epoch {epoch} - Unlogged Val loss {float(mse_unscaled)}")
                    
                
            mse = float(mse)
            history_eval.append(mse)
            print(f"Epoch {epoch} - Train loss {float(loss)} - Val loss {mse}")

            if mse < best_mse:
                best_mse = mse
                best_epoch = epoch
                best_weights = copy.deepcopy(model.state_dict())


        # restore model and return best accuracy
        model.load_state_dict(best_weights)
        print("Best Epoch: %.2f" % best_epoch)
        print("MSE: %.2f" % best_mse)
        print("RMSE: %.2f" % np.sqrt(best_mse))
        if(scale_all):
            plt.plot(history_eval_unscaled, label="Unscaled Eval loss")
        plt.plot(history_train, label="Training loss")
        plt.plot(history_eval, label="Eval loss")
        plt.title("MSE losses for scaled features and targets over epochs")
        plt.legend()
        plt.show()

    def test(scale_all=False):
        DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'
        N_EPOCHS = 495
        if(scale_all):
            LEARNING_RATE=0.001
            WD = 0.005
        else:
            LEARNING_RATE=0.001
            WD = 0.01
       

        # Load dataset and get features + target
        X, y = proc.fetch_SMRT()
        X_train, y_train, X_test, y_test = proc.split_dataset_test(X, y)

        if(scale_all):
            X_train, y_train, X_test, y_test, scaler = proc.scale_all(X_train, y_train, X_test, y_test)
        else:
            X_train, X_test = proc.scale_features(X_train, X_test)

        # Convert to 2D PyTorch tensors
        X_train = torch.tensor(X_train.values, dtype=torch.float32).to(DEVICE)
        y_train = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1).to(DEVICE)
        X_test = torch.tensor(X_test.values, dtype=torch.float32).to(DEVICE)
        y_test = torch.tensor(y_test.values, dtype=torch.float32).reshape(-1, 1).to(DEVICE)

        model = models.feedNN().to(DEVICE)

        # Define loss and optimizer
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WD)

        history_train = []
        history_eval = []
        history_eval_unscaled = []

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

            # evaluate accuracy at end of each epoch
            model.eval()
            with torch.no_grad():
                y_pred = model(X_test)
                mse = loss_fn(y_pred, y_test)
                if(scale_all):
                    y_pred_unscaled = scaler.inverse_transform(y_pred.cpu())
                    y_val_unscaled = scaler.inverse_transform(y_test.cpu())
                    y_pred_unscaled = torch.from_numpy(y_pred_unscaled).to(DEVICE)
                    y_val_unscaled = torch.from_numpy(y_val_unscaled).to(DEVICE)
                    mse_unscaled = loss_fn(y_pred_unscaled, y_val_unscaled)
                    history_eval_unscaled.append(float(mse_unscaled))
                    print(f"Epoch {epoch} - Unscaled Test loss {float(mse_unscaled)}")
                    
                
            mse = float(mse)
            r2 = r2_score(y_test.cpu(), y_pred.cpu())
            mape = mean_absolute_percentage_error(y_test.cpu(), y_pred.cpu())
            history_eval.append(mse)
            print(f"Epoch {epoch} - Train loss {float(loss)} - Test loss, MAPE and  r2: {mse, mape, r2}")

        model.eval()
        with torch.no_grad():
            y_pred_test = model(X_test)
            _, _, _, _ = mf.all_metrics(y_pred_test.cpu(), y_test.cpu())
        # Plots
        y_test_flat = y_test.cpu().numpy().flatten()
        y_pred_flat = y_pred_test.cpu().numpy().flatten()

        m, b = np.polyfit(y_test_flat, y_pred_flat, 1)

        plt.scatter(y_test_flat, y_pred_flat, c='b')
        plt.title(f"SMRT", fontsize=20)
        plt.xlabel('True tR',  fontsize=18)
        plt.ylabel('Predicted tR',  fontsize=18)
        plt.plot(y_test_flat, y_test_flat, color='lightgray', label='1:1', linewidth=2.0)
        plt.plot(y_test_flat, m*y_test_flat+b, color="red", label='predicted-true', linewidth=2.0)
        plt.legend(fontsize=15)  # Display legend
        plt.tick_params(axis='both', which='major', labelsize=14)  # Adjust tick label size
        plt.tight_layout()
        plt.show()        


        if(scale_all):
            plt.plot(history_eval_unscaled, label="Unscaled Test loss")
        plt.plot(history_train, label="Training loss")
        plt.plot(history_eval, label="Test loss")
        plt.title("MSE losses over epochs for the physicochemical SMRT dataset")
        plt.legend()
        plt.show()

    def get_final_model():
        DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'
        LEARNING_RATE=0.001
        WD = 0.01
        N_EPOCHS = 500

        # Load dataset and get features + target
        X, y = proc.fetch_SMRT()
    
        scaler = StandardScaler()
        scaler.fit(X)
        X = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)

        # Convert to 2D PyTorch tensors
        X = torch.tensor(X.values, dtype=torch.float32).to(DEVICE)
        y = torch.tensor(y.values, dtype=torch.float32).reshape(-1, 1).to(DEVICE)
        
        model = models.feedNN().to(DEVICE)
        # Define loss and optimizer
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WD)

        # Hold the best model
        best_mse = np.inf   # init to infinity
        best_epoch = 0
        best_weights = None

        start = time.time()
        for epoch in range(N_EPOCHS):
            model.train()
            # forward pass
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
            print(f"Epoch {epoch} - Loss MSE: {float(loss)}")

            mse = float(loss)
            if mse < best_mse:
                best_mse = mse
                best_epoch = epoch
                best_weights = copy.deepcopy(model.state_dict())

        end = time.time()
        print(f"Time to train: {end - start}")
        print(f"Best model at epoch {best_epoch} with MSE of {best_mse}")
        torch.save(best_weights, "SMRTModel")


if __name__=="__main__":
    
    """
    Arguments:
    ----------

    operation: String to determine what to perform (validation, testing, final)
    fine_tune and transfer_learning: Boolean values either "yes", "true", "t" or "1" strings for setting to True
    
    To launch file:

    python smrt_phys.py LPAC validation f
    """

    operation = sys.argv[1]
    scale_all = sys.argv[2]

    pipeline = SmrtModelling()

    if(operation=="validation"):
        print("Launching valiation pipeline")
        lr_ls = [0.001, 0.005, 0.01, 0.1]
        wd_ls = [0.1, 0.01, 0.005, 0]
        for lr in lr_ls:
            for wd in wd_ls:
                print(f"Validation results for LR={lr} and WD={wd}")
                pipeline.validation(lr, wd, scale_all)
    
    elif(operation=="test"):
        print("Launching testing pipeline")
        SmrtModelling.test(scale_all)
    
    elif(operation=="final"):
        print("Launching pipeline to save trained SMRT model")
        SmrtModelling.get_final_model()
    else:
        Exception("Operation not available")


