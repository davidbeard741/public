# Multivariate quantiles and long horizon forecasting with N-HiTS

[PyTorch Forecasting Docs](https://pytorch-forecasting.readthedocs.io/en/latest/tutorials/nhits.html)

[Google Colaboratory, or “Colab” for short](https://colab.research.google.com/)

```
!pip install pytorch-lightning
!pip install pytorch-forecasting
!pip install pytorch-forecasting[mqf2]
```

```
# Change the current working directory to the parent directory of the current one.
# This is done to ensure that the program is running in the expected directory.

import os   # import the os module to interact with the operating system
import warnings   # import the warnings module to handle any warnings

warnings.filterwarnings("ignore")   # suppress all warnings that might be raised

os.chdir("../../..")   # change the current working directory to the parent directory 
                       # of the parent directory of the current one.
```

```
import os
import warnings
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from pytorch_forecasting import Baseline, NHiTS, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.data.examples import generate_ar_data
from pytorch_forecasting.metrics import MAE, SMAPE, MQF2DistributionLoss, QuantileLoss
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor
```

```
# Multivariate Quantiles and Long Horizon Forecasting with N-HiTS

# Multivariate quantiles refer to estimating the joint probability distribution 
# of multiple variables, typically using a statistical model like a neural network, 
# that models the correlations between variables.

# Long horizon forecasting refers to predicting future values of a time-series 
# over a long period of time, requiring the model to capture complex temporal 
# dependencies and trends that may emerge over long periods.

# N-HiTS is a neural network architecture designed for long horizon forecasting 
# of multiple time-series. It models the dependencies between different levels 
# of the time-series and temporal dependencies within each level.

# N-HiTS can generate multivariate probabilistic forecasts and is specifically 
# designed for long horizon forecasting, capable of generating forecasts for much 
# longer periods than other models.

# N-HiTS has been shown to achieve state-of-the-art performance on a variety 
# of time-series forecasting tasks, including the M4 competition.

```

```
# Load data
# We generate a synthetic dataset to demonstrate the network’s capabilities.
# The data consists of a quadratic trend and a seasonality component.
data = generate_ar_data(seasonality=10.0, timesteps=400, n_series=100, seed=42)

# Add a static feature to the data
data["static"] = 2

# Add date information to the data for plotting and visualization
data["date"] = pd.Timestamp("2020-01-01") + pd.to_timedelta(data.time_idx, "D")

# Print the first few rows of the dataset to inspect the data structure
data.head()
```

```
# Before starting training, we need to split the dataset into a training and 
# validation TimeSeriesDataSet.

# We first define the maximum length for the encoder and prediction sequence.

max_encoder_length = 60
max_prediction_length = 20

# Then, we determine the cutoff time for the training data based on the 
# maximum prediction length.

training_cutoff = data["time_idx"].max() - max_prediction_length

context_length = max_encoder_length
prediction_length = max_prediction_length

# We create a TimeSeriesDataSet object for training data with the necessary 
# parameters like the data, target, time index, categorical encoders, and 
# group ids.

training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="value",
    categorical_encoders={"series": NaNLabelEncoder().fit(data.series)},
    group_ids=["series"],

    # In this particular implementation, only the "value" variable is 
    # considered as an unknown variable during the prediction window.

    # The N-HiTS implementation used in this code is designed to only 
    # work with the "value" variable as the unknown variable during the 
    # prediction window.

    # Therefore, no additional variables can be used in this implementation.
    time_varying_unknown_reals=["value"],
    max_encoder_length=context_length,
    max_prediction_length=prediction_length,
)

# We also specify the length of the encoder and prediction sequence for the 
# TimeSeriesDataSet object.

# We then create a validation dataset using the TimeSeriesDataSet.from_dataset 
# method, which ensures that the validation data follows the same structure as 
# the training data.

validation = TimeSeriesDataSet.from_dataset(
    training, data, min_prediction_idx=training_cutoff + 1
    )

# Finally, we create dataloaders for the training and validation datasets with 
# the specified batch size and number of workers.

batch_size = 128
train_dataloader = training.to_dataloader(
    train=True, batch_size=batch_size, num_workers=0
    )
val_dataloader = validation.to_dataloader(
    train=False, batch_size=batch_size, num_workers=0
    )
```

```
# calculate baseline absolute error
# Our baseline model predicts future values by repeating the last known value.
# The resulting SMAPE is disappointing and should not be easy to beat.
baseline_predictions = Baseline().predict(
    val_dataloader, trainer_kwargs=dict(accelerator="cpu"), return_y=True
)
SMAPE()(baseline_predictions.output, baseline_predictions.y)
```

```
# Train network

# Finding the optimal learning rate using PyTorch Lightning is easy.
# The key hyperparameter of the NHiTS model is hidden_size.

# PyTorch Forecasting is flexible enough to use NHiTS with different loss
# functions, enabling not only point forecasts but also probabilistic ones.

# Here, we will demonstrate not only a typical quantile regression but a
# multivariate quantile regression with MQF2DistributionLoss which allows
# calculation sampling consistent paths along a prediction horizon.

# This allows calculation of quantiles, e.g. of the sum over the prediction
# horizon while still avoiding the accumulating error problem from
# auto-regressive methods such as DeepAR.

# One needs to install an additional package for this quantile function:
# !pip install pytorch-forecasting[mqf2]

pl.seed_everything(42)
trainer = pl.Trainer(accelerator="cpu", gradient_clip_val=0.1)
net = NHiTS.from_dataset(
    training,
    learning_rate=3e-2,
    weight_decay=1e-2,
    loss=MQF2DistributionLoss(prediction_length=max_prediction_length),
    backcast_loss_ratio=0.0,
    hidden_size=64,
    optimizer="AdamW",
)
```

```
# instantiate a Tuner object and use it to find the optimal learning rate
# for the NHiTS model

# train_dataloader and val_dataloader are used as the training and validation
# data for the Tuner

# min_lr and max_lr specify the minimum and maximum learning rates to search
# over
res = Tuner(trainer).lr_find(
    net,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
    min_lr=1e-5,
    max_lr=1e-1,
)

# print the suggested learning rate found by the Tuner
print(f"suggested learning rate: {res.suggestion()}")

# plot the learning rate search results
fig = res.plot(show=True, suggest=True)
fig.show()

# set the learning rate of the NHiTS model to the suggested value
net.hparams.learning_rate = res.suggestion()
```

```
# Fit model

from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor

# define EarlyStopping callback with monitor on validation loss
early_stop_callback = EarlyStopping(
    monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min"
)

# define Trainer with specified settings
trainer = pl.Trainer(
    max_epochs=5,
    accelerator="cpu",
    enable_model_summary=True,
    gradient_clip_val=1.0,
    callbacks=[early_stop_callback],
    limit_train_batches=30,
    enable_checkpointing=True,
)

# define NHiTS model with specified settings
net = NHiTS.from_dataset(
    training,
    learning_rate=5e-3,
    log_interval=10,
    log_val_interval=1,
    weight_decay=1e-2,
    backcast_loss_ratio=0.0,
    hidden_size=64,
    optimizer="AdamW",
    loss=MQF2DistributionLoss(prediction_length=max_prediction_length),
)

# train the model with specified Trainer and data loaders
trainer.fit(
    net,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)
```

```
# get the best model's path from the checkpoint callback
best_model_path = trainer.checkpoint_callback.best_model_path

# load the best model from the checkpoint
best_model = NHiTS.load_from_checkpoint(best_model_path)
```

```
# We predict on the validation dataset with predict() and calculate the
# error which is well below the baseline error

predictions = best_model.predict(
    val_dataloader, trainer_kwargs=dict(accelerator="cpu"), return_y=True
)
MAE()(predictions.output, predictions.y)
```

```
# "Looking at random samples from the validation set is always a good way to
# understand if the forecast is reasonable - and it is!"

# Use your best model to make predictions on the validation set

# The `mode` argument specifies that you want raw predictions rather than
# post-processed predictions

# The `return_x` argument specifies that you want to get both the inputs and
# outputs of the model

# The `trainer_kwargs` argument specifies that you want to use the CPU rather
# than GPU for prediction

raw_predictions = best_model.predict(
    val_dataloader, mode="raw", return_x=True, trainer_kwargs=dict(accelerator="cpu")
)
```

```
# Loop through 10 examples and plot the model's predictions for each example
for idx in range(10):

    # Use the `plot_prediction` method of your best model to plot the model's
    # prediction for the current example

    # The `raw_predictions.x` argument provides the input data for the
    # current example

    # The `raw_predictions.output` argument provides the model's raw prediction
    # for the current example

    # The `idx` argument specifies the index of the example you want to plot

    # The `add_loss_to_title` argument adds the loss to the title of the plot

    best_model.plot_prediction(
        raw_predictions.x, raw_predictions.output, idx=idx, add_loss_to_title=True
    )
```

```
# Interpret model

# We can ask PyTorch Forecasting to decompose the prediction into blocks which
# focus on a different frequency spectrum, e.g. seasonality and trend with
# plot_interpretation(). This is a special feature of the NHiTS model and
# only possible because of its unique architecture. The results show that
# there seem to be many ways to explain the data and the algorithm does not
# always chooses the one making intuitive sense. This is partially down to the
# small number of time series we trained on (100). But it is also due because
# our forecasting period does not cover multiple seasonalities.

# Loop through 2 examples and plot the interpretation of the model's
# predictions for each example

# The `plot_interpretation` method of your best model decomposes the prediction
# into blocks which focus on different frequency spectra

# The `raw_predictions.x` argument provides the input data for the current
# example

# The `raw_predictions.output` argument provides the model's raw prediction
# for the current example

# The `idx` argument specifies the index of the example you want to plot

for idx in range(2):  # plot 10 examples
    best_model.plot_interpretation(raw_predictions.x, raw_predictions.output, idx=idx)
```

```
# Sampling from predictions

# Sample 500 paths from the predicted distribution of the first time series
# in the validation set
samples = best_model.loss.sample(
    raw_predictions.output["prediction"][[0]], n_samples=500
)[0]

# Plot the prediction for the first time series in the validation set
fig = best_model.plot_prediction(
    raw_predictions.x, raw_predictions.output, idx=0, add_loss_to_title=True
)
ax = fig.get_axes()[0]

# Overlay the first two sampled paths onto the plot
ax.plot(samples[:, 0], color="g", label="Sample 1")
ax.plot(samples[:, 1], color="r", label="Sample 2")

# Add a legend to the plot indicating the color and label of each line
fig.legend()
```

```
# As expected, the variance of predictions is smaller within each sample
# than accross all samples.

# Compute the variance of all the samples
print(f"Var(all samples) = {samples.var():.4f}")

# Compute the mean variance across all samples for each time step
print(f"Mean(Var(sample)) = {samples.var(0).mean():.4f}")
```

```
# We can now do something new and plot the distribution of sums of forecasts
# over the entire prediction range.

# Compute the sum of predictions across all samples for each time step,
# and convert to a NumPy array
sums = samples.sum(0).numpy()

# Plot a histogram of the sum of predictions
plt.hist(sums, bins=30)

# Set the label for the x-axis
plt.xlabel("Sum of predictions")

# Set the label for the y-axis
plt.ylabel("Frequency")
```
