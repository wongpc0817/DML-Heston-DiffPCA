#!/usr/bin/env python
# coding: utf-8

# In[14]:


import tensorflow as tf2
assert tf2.__version__ >= "2.0"
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import time
from tqdm import tqdm_notebook
import warnings
warnings.filterwarnings('ignore')
import pandas as pd 
from tqdm import tqdm
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
import random
import os
from scipy.integrate import quad_vec  # quad_vec allows to compute integrals accurately
from scipy.stats import norm
from scipy.stats import qmc # to perform Latin Hypercube Sampling (LHS) 
import pandas as pd 

def set_random_seed(seed=42):
    tf.compat.v1.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING logs
tf = tf2.compat.v1
tf.logging.set_verbosity(tf.logging.ERROR)
set_random_seed()
real_type = tf.float32

## Load the dataset
feller_data = pd.read_csv('data//feller_d2.csv')

## Obtain 2nd order differentials and their PCs
second_differentials = feller_data.iloc[:,-36:]
normalized_zscore = (second_differentials - second_differentials.mean()) / second_differentials.std()
pca_d2 = PCA(n_components=15)
second_differential_label = pca_d2.fit_transform(normalized_zscore[:9000])

## Train-test Split (90-10)
feller_testing = feller_data.iloc[9000:]
feller_data = feller_data.iloc[:9000]

training_col = feller_data.columns[:8]
training_target = feller_data.columns[9]
network_inputs = feller_data[training_col].values
network_outputs = feller_data[training_target].values

func_names = ["lm", "r", "tau", "theta", "sigma", "rho", "kappa", "v0"]
network_inputs = feller_data[func_names]
option_prices = feller_data['P_hat']
network_first_order = feller_data[[f"diff wrt {col}" for col in func_names]].values
sec_order_names = []
for i in func_names:
    for j in func_names:
        if os.path.exists(f"data//d2_{i}_{j}.csv"):
            sec_order_names.append(f"d2_{i}_{j}")
network_second_order = feller_data[[f"{col}" for col in sec_order_names]].values


# <center>
# <img src="second_order_differential.png" width=700>
# </center>

# We consider four models:
# - Model 1: The benchmark framework 
#     
#     (Heston parameters $\mapsto$ option price)
# - Model 2: Model trained with 1st order differentials 
# 
#     (Heston parameters $\mapsto$ (option price & 1st order differentials))
# - Model 3: Model trained with 1st and 2nd order differentials 
# 
#     (Heston parameters $\mapsto$ (option price & 1st and 2nd differentials))
# - Model 4: Model trained with 1st order differentials, and 2nd order Diff-PCA differentials 
# 
#     (Heston parameters $\mapsto$ (option price & 1st order differentials, 2nd order Diff-PCA differentials))

# Model 1: Without Differentials

# In[15]:


def twin_net_with_first_order(hidden_units=64, hidden_layers=3):
    raw_inputs = tf.keras.Input(shape=(8,))
    x = raw_inputs
    # Hidden layers
    for _ in range(hidden_layers):
        x = tf.keras.layers.Dense(hidden_units, activation='softplus')(x)
    # Output layer (option price)
    option_price = tf.keras.layers.Dense(1)(x)  # Predicted option price
    # Create model with two outputs: option price and second-order differential
    model = tf.keras.Model(inputs=raw_inputs, outputs=[option_price])
    return model

# Function to compute gradients and return price, first-order, and second-order differentials
def compute_grad_model1(model, raw_inputs):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(raw_inputs)  # Watch the raw inputs
        option_price = model(raw_inputs)  # Get the outputs (option price, second-order diff)
    
    # Compute the first-order differential (gradient of option price w.r.t raw inputs)
    first_order_differential = tape.gradient(option_price, raw_inputs)
    
    return option_price, first_order_differential

def compute_first_order_differential(model, raw_inputs):
    with tf.GradientTape() as tape:
        tape.watch(raw_inputs)  # Watch the raw inputs
        option_price = model(raw_inputs)[0]  # Get the option price (y_pred[0])
    
    # Compute the first-order differential (gradients w.r.t raw inputs)
    first_order_differential = tape.gradient(option_price, raw_inputs)
    
    return first_order_differential

# Loss function that uses true gradients and predicted gradients
def loss_fn(y_true, y_pred, true_first_order_differentials):
    # Extract the true and predicted option price and second-order differentials
    true_option_price = y_true[0]  # The first element contains the true option price
    pred_option_price = y_pred[0]  # The first element contains the predicted option price
    pred_first_order_diff = y_pred[1]

    # Option price loss (L2 loss)
    price_loss = tf.reduce_mean(tf.square(true_option_price - pred_option_price))  
    # First-order differential loss (L2 loss)
    first_order_loss = tf.reduce_mean(tf.square(true_first_order_differentials - pred_first_order_diff))

    # Total loss (could be a weighted sum of the individual losses)
    total_loss = price_loss + 0.5 * first_order_loss 

    # Return all losses (can be used for monitoring during training)
    return total_loss, price_loss, first_order_loss


# In[16]:


# Define optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Generate some random input data (raw inputs)
raw_inputs = tf.convert_to_tensor(network_inputs, dtype=tf.float32)

# Generate some dummy true data for loss calculation (real data would come from a model/simulation)
true_prices = tf.convert_to_tensor(option_prices.values.reshape(1,-1), dtype=tf.float32)
true_first_order_differentials = tf.convert_to_tensor(network_first_order, dtype=tf.float32)
true_second_order_differentials = tf.convert_to_tensor(network_second_order, dtype=tf.float32)

# Create the model
model = twin_net_with_first_order()
history_1 = {
    'total_loss': [],
    'price_loss': [],
    'first_order_loss': [],
    'second_order_loss': []
}

# Training loop
start_time1 = time.time()
for epoch in range(1000):  # Number of epochs
    with tf.GradientTape() as tape:
        # Get predictions (tuple of price, first-order diff, second-order diff)
        predicted_price, predicted_first_order = compute_grad_model1(model, raw_inputs)
        
        # Compute loss (pass the true gradients as part of the loss function)
        total_loss, price_loss, first_order_loss = loss_fn(
            [true_prices, true_second_order_differentials],  # True data
            [predicted_price, predicted_first_order],  # Model predictions
            true_first_order_differentials  # True first-order differentials
        )
    
    # Compute gradients with respect to model parameters
    gradients = tape.gradient(price_loss, model.trainable_variables)
    
    # Update model parameters using the optimizer
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    history_1['total_loss'].append(total_loss.numpy())
    history_1['price_loss'].append(price_loss.numpy())
    history_1['first_order_loss'].append(first_order_loss.numpy())
    
    # Print the loss every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch {epoch}:")
        print(f"  Total Loss: {total_loss.numpy()}")
        print(f"  Price Loss: {price_loss.numpy()}")
        print(f"  First-Order Loss: {first_order_loss.numpy()}")
end_time1 = time.time()
model.save(f"models//model1.h5")


# Model 2: Training with first order differentials

# In[17]:


def twin_net_with_first_order(hidden_units=64, hidden_layers=3):
    raw_inputs = tf.keras.Input(shape=(8,))
    x = raw_inputs
    # Hidden layers
    for _ in range(hidden_layers):
        x = tf.keras.layers.Dense(hidden_units, activation='softplus')(x)
    
    # Output layer (option price)
    option_price = tf.keras.layers.Dense(1)(x)  # Predicted option price
    
    # Create model with two outputs: option price and second-order differential
    model = tf.keras.Model(inputs=raw_inputs, outputs=[option_price])
    return model

# Function to compute gradients and return price, first-order, and second-order differentials
def compute_grad_model2(model, raw_inputs):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(raw_inputs)  # Watch the raw inputs
        option_price = model(raw_inputs)  # Get the outputs (option price, second-order diff)
    
    # Compute the first-order differential (gradient of option price w.r.t raw inputs)
    first_order_differential = tape.gradient(option_price, raw_inputs)
    
    return option_price, first_order_differential

def compute_first_order_differential(model, raw_inputs):
    with tf.GradientTape() as tape:
        tape.watch(raw_inputs)  # Watch the raw inputs
        option_price = model(raw_inputs)[0]  # Get the option price (y_pred[0])
    
    # Compute the first-order differential (gradients w.r.t raw inputs)
    first_order_differential = tape.gradient(option_price, raw_inputs)
    
    return first_order_differential
# Loss function that uses true gradients and predicted gradients
def loss_fn(y_true, y_pred, true_first_order_differentials, lambda1):
    # Extract the true and predicted option price and second-order differentials
    true_option_price = y_true[0]  # The first element contains the true option price
    pred_option_price = y_pred[0]  # The first element contains the predicted option price
    pred_first_order_diff = y_pred[1]

    # Option price loss (L2 loss)
    price_loss = tf.reduce_mean(tf.square(true_option_price - pred_option_price))  
    # First-order differential loss (L2 loss)
    first_order_loss = tf.reduce_mean(tf.square(true_first_order_differentials - pred_first_order_diff))
    # Total loss (could be a weighted sum of the individual losses)
    total_loss = price_loss + lambda1 * first_order_loss 
    
    # Return all losses (can be used for monitoring during training)
    return total_loss, price_loss, first_order_loss


# In[18]:


# Define optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Generate some random input data (raw inputs)
raw_inputs = tf.convert_to_tensor(network_inputs, dtype=tf.float32)

# Generate some dummy true data for loss calculation (real data would come from a model/simulation)
true_prices = tf.convert_to_tensor(option_prices.values.reshape(1,-1), dtype=tf.float32)
true_first_order_differentials = tf.convert_to_tensor(network_first_order, dtype=tf.float32)
true_second_order_differentials = tf.convert_to_tensor(network_second_order, dtype=tf.float32)

# Create the model
model = twin_net_with_first_order()
history_2 = {
    'total_loss': [],
    'price_loss': [],
    'first_order_loss': [],
    'second_order_loss': [],
    'lambda1': [],
}
start_time2 = time.time()
# Training loop
for lambda1 in [0.1,0.5,0.7]:
    for epoch in range(1000):  # Number of epochs
        with tf.GradientTape() as tape:
            # Get predictions (tuple of price, first-order diff, second-order diff)
            predicted_price, predicted_first_order = compute_grad_model2(model, raw_inputs)
            
            # Compute loss (pass the true gradients as part of the loss function)
            total_loss, price_loss, first_order_loss = loss_fn(
                [true_prices, true_second_order_differentials],  # True data
                [predicted_price, predicted_first_order],  # Model predictions
                true_first_order_differentials,  # True first-order differentials
                lambda1
            )
        
        # Compute gradients with respect to model parameters
        gradients = tape.gradient(total_loss, model.trainable_variables)
        
        # Update model parameters using the optimizer
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        history_2['total_loss'].append(total_loss.numpy())
        history_2['price_loss'].append(price_loss.numpy())
        history_2['first_order_loss'].append(first_order_loss.numpy())
        history_2['lambda1'].append(lambda1)
        # Print the loss every 10 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch}:")
            print(f"  Total Loss: {total_loss.numpy()}")
            print(f"  Price Loss: {price_loss.numpy()}")
            print(f"  First-Order Loss: {first_order_loss.numpy()}")
    model.save(f"models//model2_{int(lambda1*10)}.h5")
end_time2 = time.time()


# Model 3: Training with second order differentials without Diff-PCA

# In[19]:


def twin_net_with_first_second_order(input_dim, hidden_units=64, hidden_layers=3):
    raw_inputs = tf.keras.Input(shape=(8,))
    x = raw_inputs

    # Hidden layers
    for _ in range(hidden_layers):
        x = tf.keras.layers.Dense(hidden_units, activation='softplus')(x)
    
    # Output layer (option price)
    option_price = tf.keras.layers.Dense(1)(x)  # Predicted option price
    second_order_diff = tf.keras.layers.Dense(input_dim)(x)  # Predicted second-order differential (36 elements)
    
    # Create model with two outputs: option price and second-order differential
    model = tf.keras.Model(inputs=raw_inputs, outputs=[option_price, second_order_diff])
    return model

# Function to compute gradients and return price, first-order, and second-order differentials
def compute_grad_model3(model, raw_inputs):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(raw_inputs)  # Watch the raw inputs
        option_price, second_order_diff = model(raw_inputs)  # Get the outputs (option price, second-order diff)
    
    # Compute the first-order differential (gradient of option price w.r.t raw inputs)
    first_order_differential = tape.gradient(option_price, raw_inputs)
    
    return option_price, first_order_differential, second_order_diff

def compute_first_order_differential(model, raw_inputs):
    with tf.GradientTape() as tape:
        tape.watch(raw_inputs)  # Watch the raw inputs
        option_price = model(raw_inputs)[0]  # Get the option price (y_pred[0])
    
    # Compute the first-order differential (gradients w.r.t raw inputs)
    first_order_differential = tape.gradient(option_price, raw_inputs)
    
    return first_order_differential
# Loss function that uses true gradients and predicted gradients
def loss_fn(y_true, y_pred, true_first_order_differentials, lambda1, lambda2):
    # Extract the true and predicted option price and second-order differentials
    true_option_price = y_true[0]  # The first element contains the true option price
    true_second_order_diff = y_true[1]  # The remaining elements contain the true second-order differentials
    pred_option_price = y_pred[0]  # The first element contains the predicted option price
    pred_second_order_diff = y_pred[2]  # The second element contains the predicted second-order differentials
    pred_first_order_diff = y_pred[1]
    # Option price loss (L2 loss)
    price_loss = tf.reduce_mean(tf.square(true_option_price - pred_option_price))  
    # Second-order differential loss (L2 loss)
    second_order_loss = tf.reduce_mean(tf.square(true_second_order_diff - pred_second_order_diff))
    
    # First-order differential loss (L2 loss)
    first_order_loss = tf.reduce_mean(tf.square(true_first_order_differentials - pred_first_order_diff))

    # Total loss (could be a weighted sum of the individual losses)
    total_loss = price_loss + lambda1 * first_order_loss + lambda2* second_order_loss
    
    # Return all losses (can be used for monitoring during training)
    return total_loss, price_loss, first_order_loss, second_order_loss


# In[20]:


# Define optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Generate some random input data (raw inputs)
raw_inputs = tf.convert_to_tensor(network_inputs, dtype=tf.float32)

# Generate some dummy true data for loss calculation (real data would come from a model/simulation)
true_prices = tf.convert_to_tensor(option_prices.values.reshape(1,-1), dtype=tf.float32)
true_first_order_differentials = tf.convert_to_tensor(network_first_order, dtype=tf.float32)
true_second_order_differentials = tf.convert_to_tensor(network_second_order, dtype=tf.float32)

# Create the model
model = twin_net_with_first_second_order(network_second_order.shape[1])
history_3 = {
    'total_loss': [],
    'price_loss': [],
    'first_order_loss': [],
    'second_order_loss': [],
    'lambda1': [],
    'lambda2': []
}
start_time3 = time.time()
# Training loop
for lambda1 in [0.1,0.5,0.7]:
    for lambda2 in [0.1,0.5,0.7]:
        for epoch in range(1000):  # Number of epochs
            with tf.GradientTape() as tape:
                # Get predictions (tuple of price, first-order diff, second-order diff)
                predicted_price, predicted_first_order, predicted_second_order = compute_grad_model3(model, raw_inputs)
                
                # Compute loss (pass the true gradients as part of the loss function)
                total_loss, price_loss, first_order_loss, second_order_loss = loss_fn(
                    [true_prices, true_second_order_differentials],  # True data
                    [predicted_price, predicted_first_order, predicted_second_order],  # Model predictions
                    true_first_order_differentials,  # True first-order differentials
                    lambda1, lambda2
                )
            
            # Compute gradients with respect to model parameters
            gradients = tape.gradient(total_loss, model.trainable_variables)
            
            # Update model parameters using the optimizer
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            history_3['total_loss'].append(total_loss.numpy())
            history_3['price_loss'].append(price_loss.numpy())
            history_3['first_order_loss'].append(first_order_loss.numpy())
            history_3['second_order_loss'].append(second_order_loss.numpy())
            history_3['lambda1'].append(lambda1)
            history_3['lambda2'].append(lambda2)
            # Print the loss every 10 epochs
            if epoch % 100 == 0:
                print(f"Epoch {epoch}:")
                print(f"  Total Loss: {total_loss.numpy()}")
                print(f"  Price Loss: {price_loss.numpy()}")
                print(f"  First-Order Loss: {first_order_loss.numpy()}")
                print(f"  Second-Order Loss: {second_order_loss.numpy()}")
        model.save(f"models//model3_{int(lambda1*10)}_{int(lambda2*10)}.h5")

end_time3 = time.time()


# Model 4: Training with second order differentials with Diff-PCA

# In[21]:


def twin_net_with_first_second_order(input_dim, hidden_units=64, hidden_layers=3):
    raw_inputs = tf.keras.Input(shape=(8,))
    x = raw_inputs

    # Hidden layers
    for _ in range(hidden_layers):
        x = tf.keras.layers.Dense(hidden_units, activation='softplus')(x)
    
    # Output layer (option price)
    option_price = tf.keras.layers.Dense(1)(x)  # Predicted option price
    second_order_diff = tf.keras.layers.Dense(input_dim)(x)  # Predicted second-order differential (36 elements)
    
    # Create model with two outputs: option price and second-order differential
    model = tf.keras.Model(inputs=raw_inputs, outputs=[option_price, second_order_diff])
    return model

# Function to compute gradients and return price, first-order, and second-order differentials
def compute_grad_model4(model, raw_inputs):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(raw_inputs)  # Watch the raw inputs
        option_price, second_order_diff = model(raw_inputs)  # Get the outputs (option price, second-order diff)
    
    # Compute the first-order differential (gradient of option price w.r.t raw inputs)
    first_order_differential = tape.gradient(option_price, raw_inputs)
    
    return option_price, first_order_differential, second_order_diff

def compute_first_order_differential(model, raw_inputs):
    with tf.GradientTape() as tape:
        tape.watch(raw_inputs)  # Watch the raw inputs
        option_price = model(raw_inputs)[0]  # Get the option price (y_pred[0])
    
    # Compute the first-order differential (gradients w.r.t raw inputs)
    first_order_differential = tape.gradient(option_price, raw_inputs)
    
    return first_order_differential
# Loss function that uses true gradients and predicted gradients
def loss_fn(y_true, y_pred, true_first_order_differentials, lambda1, lambda2):
    # Extract the true and predicted option price and second-order differentials
    true_option_price = y_true[0]  # The first element contains the true option price
    true_second_order_diff = y_true[1]  # The remaining elements contain the true second-order differentials
    pred_option_price = y_pred[0]  # The first element contains the predicted option price
    pred_second_order_diff = y_pred[2]  # The second element contains the predicted second-order differentials
    pred_first_order_diff = y_pred[1]
    # Option price loss (L2 loss)
    price_loss = tf.reduce_mean(tf.square(true_option_price - pred_option_price))  
    
    # print(true_second_order_diff,pred_second_order_diff)
    # Second-order differential loss (L2 loss)
    second_order_loss = tf.reduce_mean(tf.square(true_second_order_diff - pred_second_order_diff))
    
    # First-order differential loss (L2 loss)
    first_order_loss = tf.reduce_mean(tf.square(true_first_order_differentials - pred_first_order_diff))
    # Total loss (could be a weighted sum of the individual losses)
    total_loss = price_loss + lambda1 * first_order_loss + lambda2*second_order_loss
    
    # Return all losses (can be used for monitoring during training)
    return total_loss, price_loss, first_order_loss, second_order_loss


# In[22]:


# Define optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Generate some random input data (raw inputs)
raw_inputs = tf.convert_to_tensor(network_inputs, dtype=tf.float32)

# Generate some dummy true data for loss calculation (real data would come from a model/simulation)
true_prices = tf.convert_to_tensor(option_prices.values.reshape(1,-1), dtype=tf.float32)
true_first_order_differentials = tf.convert_to_tensor(network_first_order, dtype=tf.float32)
true_second_order_differentials = tf.convert_to_tensor(second_differential_label, dtype=tf.float32)

# Create the model
model = twin_net_with_first_second_order(second_differential_label.shape[1])
history_4 = {
    'total_loss': [],
    'price_loss': [],
    'first_order_loss': [],
    'second_order_loss': [],
    'lambda1': [],
    'lambda2': []
}

start_time4 = time.time()
# Training loop
for lambda1 in [0.1,0.5,0.7]:
    for lambda2 in [0.1,0.5,0.7]:
        for epoch in range(1000):  # Number of epochs
            with tf.GradientTape() as tape:
                # Get predictions (tuple of price, first-order diff, second-order diff)
                predicted_price, predicted_first_order, predicted_second_order = compute_grad_model4(model, raw_inputs)
                
                # Compute loss (pass the true gradients as part of the loss function)
                total_loss, price_loss, first_order_loss, second_order_loss = loss_fn(
                    [true_prices, true_second_order_differentials],  # True data
                    [predicted_price, predicted_first_order, predicted_second_order],  # Model predictions
                    true_first_order_differentials,  # True first-order differentials
                    lambda1, lambda2
                )
            
            # Compute gradients with respect to model parameters
            gradients = tape.gradient(total_loss, model.trainable_variables)
            
            # Update model parameters using the optimizer
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            history_4['total_loss'].append(total_loss.numpy())
            history_4['price_loss'].append(price_loss.numpy())
            history_4['first_order_loss'].append(first_order_loss.numpy())
            history_4['second_order_loss'].append(second_order_loss.numpy())
            history_4['lambda1'].append(lambda1)
            history_4['lambda2'].append(lambda2)
            # Print the loss every 10 epochs
            if epoch % 100 == 0:
                print(f"Epoch {epoch}:")
                print(f"  Total Loss: {total_loss.numpy()}")
                print(f"  Price Loss: {price_loss.numpy()}")
                print(f"  First-Order Loss: {first_order_loss.numpy()}")
                print(f"  Second-Order Loss: {second_order_loss.numpy()}")
        model.save(f"models//model4_{int(lambda1*10)}_{int(lambda2*10)}.h5")

end_time4 = time.time()


# Save all the training records.

# In[23]:


dataframes = []
for idx, history in enumerate([history_1,history_2,history_3,history_4], start=1):
    max_length = max(len(v) for v in history.values() if v)

    # Fill empty lists with NaNs
    for key, value in history.items():
        if len(value) < max_length:
            history[key] = value + [np.nan] * (max_length - len(value))
    history['index'] = list(range(len(next(iter(history.values())))))

    # Convert to DataFrame
    df = pd.DataFrame(history)
    df['source'] = f'history{idx}'  # Add a column identifying the source
    dataframes.append(df)

# Combine all DataFrames into one
combined_df = pd.concat(dataframes, ignore_index=True)

# Print the resulting DataFrame
print(combined_df)
combined_df.to_csv("results//learning_history.csv", index=False)


# In[31]:


print("Training Time")
print(f"Model 1: {end_time1 - start_time1:f}")
print(f"Model 2: {(end_time2 - start_time2)/3:f}")
print(f"Model 3: {(end_time3 - start_time3)/9:f}")
print(f"Model 4: {(end_time4 - start_time4)/9:f}")

