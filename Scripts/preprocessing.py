import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

##### Fetching dataset methods #####

def fetch_SMRT():
    data = pd.read_csv("data\physicochemical_bigData.csv", header=0, skiprows=[0], index_col=0)
    data.drop("smiles", axis=1, inplace=True)
    data.dropna(axis=0, inplace=True)

    y = data.tR
    X = data.drop("tR", axis=1)
    
    return X, y

def fetch_lpac_data():
    data = pd.read_csv("data\physicochemical_small_Data.csv", header=0, skiprows=[0], index_col=0)
    extraColumns = ["BCUT2D_MWHI", "BCUT2D_MWLOW", "BCUT2D_CHGHI", "BCUT2D_CHGLO", "BCUT2D_LOGPHI", "BCUT2D_LOGPLOW", "BCUT2D_MRHI", "BCUT2D_MRLOW"]
    data.drop(extraColumns, axis=1, inplace=True)

    y = data[["tR_2_20", "tR_3.5_20", "tR_5_20", "tR_6.5_20", "tR_8_20"]]
    X = data.drop(["tR_2_20", "tR_3.5_20", "tR_5_20", "tR_6.5_20", "tR_8_20"], axis=1)
    
    return X, y

def fetch_ACN_data():
    data = pd.read_csv("data\MT_common_agneshka_ACN.csv", header=0, index_col=0)
    extraColumns =['Mass', 'BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BCUT2D_CHGHI', 'BCUT2D_CHGLO', 'BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW', 'BCUT2D_MRHI', 'BCUT2D_MRLOW']
    data.drop(extraColumns, axis=1, inplace=True)
    data.dropna(axis=0, inplace=True)

    y = data[["tR_2.5_ACN", "tR_3.3_ACN", "tR_4.9_ACN", "tR_6.8_ACN", "tR_8.9_ACN"]]
    X = data.drop(["Name", "tR_2.5_ACN", "tR_3.3_ACN", "tR_4.9_ACN", "tR_6.8_ACN", "tR_8.9_ACN"], axis=1)
    
    return X, y


def fetch_riken_data():
    data = pd.read_csv("data\train_molecular_descriptors.csv", header=0, index_col=0)
    extraColumns = ["BCUT2D_MWHI", "BCUT2D_MWLOW", "BCUT2D_CHGHI", "BCUT2D_CHGLO", "BCUT2D_LOGPHI", "BCUT2D_LOGPLOW", "BCUT2D_MRHI", "BCUT2D_MRLOW"]
    data.drop(extraColumns, axis=1, inplace=True)

    y = data["tR"]
    X = data.drop(["Name", "tR"], axis=1)

    return X, y

###### Splitting and scaling methods ######

def split_dataset_val(X,y, validation_set=0.10, test_size=0.10):
    """
    Splits the initial dataset X and y into a train, validation and test sets. 

    Parameters
    ----------
    X: dataset features
    y: dataset target
    validation_set: size of validation set
    test_size: size of test set

    Returns
    --------

    Training, validation and test sets, separated into features and target
    """
    X_train_val, X_test, y_train_val, y_test = train_test_split(X,y, test_size=test_size, shuffle=True, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=validation_set, shuffle=True)

    return X_train, y_train, X_val, y_val, X_test, y_test

def split_dataset_test(X,y, test_size=0.10):
    """
    Splits the initial dataset X and y into a train and test 

    Parameters
    ----------
    X: dataset features
    y: dataset target
    test_size: size of test set

    Returns
    --------

    Training and test sets, separated into features and target
    """
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=test_size, shuffle=True, random_state=0)
    return X_train, y_train, X_test, y_test

def scale_features(X_train, X_val):
    """
    Scales 2 input datasets, based on the mean and variance of the first one.

    Parameters
    ----------
    X_train: datasets on which mean and variance are computed
    X_val: second dataset
    Returns
    --------

    Both input datasets, but scaled
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train =  pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
    X_val =  pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)

    return X_train, X_val


def scale_all(X_train, y_train, X_val, y_val):
    """
    Scales 2 input datasets (features and targets), based on the mean and variance of the first one.

    Parameters
    ----------
    X_train: datasets on which mean and variance are computed
    y_train: first dataset targets
    X_val: second dataset
    y_val: seconda dataset targets
    Returns
    --------

    Both input datasets (features and targets), but scaled
    """

    scaler_features = StandardScaler()
    scaler_target = StandardScaler()
    scaler_features.fit(X_train)
    scaler_target.fit(y_train.values.reshape(-1,1))

    X_train = pd.DataFrame(scaler_features.transform(X_train), columns=X_train.columns, index=X_train.index)
    X_val = pd.DataFrame(scaler_features.transform(X_val), columns=X_val.columns, index=X_val.index)
    y_train = scaler_target.transform(y_train.values.reshape(-1,1))
    y_train = pd.Series(y_train.flatten())
    y_val = scaler_target.transform(y_val.values.reshape(-1,1))
    y_val = pd.Series(y_val.flatten())
    

    return X_train, y_train, X_val, y_val, scaler_target