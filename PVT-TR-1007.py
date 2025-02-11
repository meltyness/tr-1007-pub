# Python imports
import os
import datetime
import gc
from multiprocessing.pool import ThreadPool
import multiprocessing as mp
import logging
import sys
import pickle
import io
import time

# Data-science Specific imports
import numpy as np
import pandas as pd
import tensorflow as tf
#import elasticsearch as es
#import elasticsearch.helpers

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

# Viz imports
import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import math

# Backtesting library
from simple_back.backtester import BacktesterBuilder

DEBUGGING_ENABLED = True

def debug_dump(str):
    if DEBUGGING_ENABLED:
        print(str)

TARGET_TICKER = sys.argv[1]

df = pd.read_pickle("./BIG.pickle")

TARGET_TICKER = "Close_" + TARGET_TICKER
print("processing ", TARGET_TICKER)

if not TARGET_TICKER in df.columns:
    debug_dump("not present in corpus")
    #continue

debug_dump("counting trade days to first trade")
i = 0
for v in df[TARGET_TICKER]:
    if v > 0:
        break
    i+=1

days_traded = (df[TARGET_TICKER] > 0).replace(to_replace=False, value=np.NaN).count()

debug_dump("setting up main constraints")
#Establish main constants

n = len(df)
CONV_WIDTH = 40
fuck_the_dog = False
deterministic_fourier_features = True
BATCH_SIZE = 1
SEARCH_WIDTH = 1
TRAINING_SPLIT = [0.9, 0.95]
#MINIMUM_DAYS = np.min([3750, n-i])
MINIMUM_DAYS = np.min([3750,n-i,days_traded])
INITIALIZATION_WIDTH = 0.05
GLOBAL_MODELING_PATIENCE = 4

if MINIMUM_DAYS * (TRAINING_SPLIT[1]-TRAINING_SPLIT[0]) < CONV_WIDTH:
    print("Can't validate, skipping modeling of " + TARGET_TICKER)
    exit()

debug_dump("filtering to close and vol")
mask = df.columns.str.contains('Close.*|v_.*')

df = df.loc[:,mask]

mask = df.columns.str.contains('Close_[^_]*$')

debug_dump("rapture tickers that haven't existed long enough")
test = (((df.loc[:,mask] > 0).replace(to_replace=False, value=np.NaN).count()) >= MINIMUM_DAYS)

rapture = []

for col, good_enough in test.iteritems():
    alt_col = col.replace("Close_","v_")
    if good_enough:
        rapture += [df[col]]
        rapture += [df[alt_col]]

df = pd.DataFrame(rapture).transpose()

del rapture

debug_dump("normalize volumes to prevent float overflow to Inf")
mask = df.columns.str.contains('v_.*')

for col in df.loc[:,mask]:
    df[col] = df[col].div(df[col].max())

#mask = df.columns.str.contains("Close_[^_]+$")
#mask2 = df.columns.str.contains(TARGET_TICKER)

#for col in df.loc[:,(mask & ~mask2)]:
#    df[col] = df[col].div(df[col].max())

date_time = pd.to_datetime(df.index, format='%d.%m.%Y %H:%M:%S')

timestamp_s = date_time.map(datetime.datetime.timestamp)

day = 24*60*60
month = (30.42)*day
year = (365.2425)*day

# Append consistent features representing periodicity for learning
df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))

df['Month sin'] = np.sin(timestamp_s * (2 * np.pi / month))
df['Month cos'] = np.cos(timestamp_s * (2 * np.pi / month))

df['Quarterly sin'] = np.sin(timestamp_s * (2 * np.pi / (3*month)))
df['Quarterly cos'] = np.cos(timestamp_s * (2 * np.pi / (3*month)))

df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

df['Super sin'] = np.sin(timestamp_s * (2 * np.pi / (15*year)))
df['Super cos'] = np.cos(timestamp_s * (2 * np.pi / (15*year)))

debug_dump("generate random fourier features")
# RFF-generator
RFF_DOF = int(np.floor( 8 * 5188 / df.shape[1] ))
print(RFF_DOF)
RFF_SMALLNESS_PREFERENCE = 1 # a larger value means a preference for lower frequencies
RFF_MAX = 2560 * 4

if deterministic_fourier_features:
    features = (np.linspace(2, RFF_MAX, num=RFF_DOF) ** RFF_SMALLNESS_PREFERENCE).tolist()
else:
    features = (RFF_MAX * np.random.rand(1, RFF_DOF) ** RFF_SMALLNESS_PREFERENCE).tolist()[0]

# Badly understood dollar value multiplier -- may enable better adherance to long-term gains
VOLATILITY_COST = 1

collected_dfs = []
mask = df.columns.str.contains('Close.*')
debug_dump("building a thread pool to augment features")
def _sinful_thread(col):
    global collected_dfs
    for iteration, w in enumerate(features):
        #new_col = df[col].transform(lambda x: VOLATILITY_COST * np.sin(w * x, dtype=np.float16))
        new_col = VOLATILITY_COST * np.sin(w * df[col])
        new_col.name = col + "_f_s_" + str(iteration)
        collected_dfs += [new_col]
        #new_col = df[col].transform(lambda x: VOLATILITY_COST * np.cos(w * x, dtype=np.float16))
        new_col = VOLATILITY_COST * np.cos(w * df[col])
        new_col.name = col + "_f_c_" + str(iteration)
        collected_dfs += [new_col]
        
debug_dump("threads starting")

pool = ThreadPool(processes=16)    
res = [pool.apply_async(_sinful_thread,args=(col,)) for col in df.loc[:,mask]]
pool.close()
pool.join()

debug_dump("threads finished")

debug_dump("forcing float32 datatype")
for c in collected_dfs:
    c = c.astype(np.float32)

debug_dump("appending augmenting features")
df = df.join(collected_dfs)

if fuck_the_dog:
    debug_dump("WARNING: downsampling usually breaks training")
    df = df.astype(np.float16)

del collected_dfs

# old style: now basing this off of discovered size
n = len(df)

i = 0
for v in df[TARGET_TICKER]:
    if v > 0:
        break
    i+=1

debug_dump("performing training split")

train_df = df[i:int((n-i)*TRAINING_SPLIT[0])+i]
val_df = df[int((n-i)*TRAINING_SPLIT[0])+i:int((n-i)*TRAINING_SPLIT[1])+i]
test_df = df[int((n-i)*TRAINING_SPLIT[1])+i:]

num_features = df.shape[1]

column_indices = {name: i for i, name in enumerate(df.columns)}

class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                 train_df=train_df, val_df=val_df, test_df=test_df,
                 label_columns=None):
        #Store raw data
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                         enumerate(label_columns)}

        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        #Work out the window parameters
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return 'n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns], axis=-1)

        #repair shape information
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=BATCH_SIZE,)

        ds = ds.map(self.split_window)

        return ds

    def plot(self, model=None, plot_col='T (degC)', max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 16))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
              label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
              label_col_index = plot_col_index

            if label_col_index is None:
              continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
              predictions = model(inputs)
              plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                          marker='X', edgecolors='k', label='Predictions',
                          c='#ff7f0e', s=64)

            if n == 0:
              plt.legend()

        plt.xlabel('Time [h]')

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of 'inputs, labels' for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get from .train dataset
            result = next(iter(self.val))
            # and cache it
            self._example = result

        return result

debug_dump("training simple baseline model")
single_step_window = WindowGenerator(
    input_width=24, label_width=1, shift=24,
    label_columns=[TARGET_TICKER])

class Baseline(tf.keras.Model):
    def __init__(self, label_index=None):
        super().__init__()
        self.label_index = label_index

    def call(self, inputs):
        if self.label_index is None:
            return inputs
        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]

baseline = Baseline(label_index=column_indices[TARGET_TICKER])

baseline.compile(loss=tf.losses.MeanSquaredError(),
                 metrics=[tf.metrics.MeanAbsoluteError()])

val_performance = {}
performance = {}

val_performance['Baseline'] = baseline.evaluate(single_step_window.val, verbose=0)
performance['Baseline'] = baseline.evaluate(single_step_window.test, 
                                           verbose=0)

MAX_EPOCHS = 10000
def compile_and_fit(model, window, patience=GLOBAL_MODELING_PATIENCE):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min',
                                                      restore_best_weights=True)

    #TODO: Make the learning rate contingent on the weight of the input parameters
    initial_learning_rate = 0.0000375
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=3654,
        decay_rate=0.9,
        staircase=True)

    model.compile(loss=tf.losses.MeanSquaredError(),
                  #optimizer=tf.optimizers.SGD(momentum=4e-3, learning_rate=7.5e-8),
                  #optimizer=tf.optimizers.SGD(momentum=0.00006, learning_rate=0.0025),
                  optimizer=tf.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e0),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=MAX_EPOCHS,
                        validation_data=window.val, workers=16,
                        callbacks=[early_stopping], verbose=2)

    print(model.summary())
    return history

conv_window = WindowGenerator(
    input_width=CONV_WIDTH,
    label_width=1,
    shift=1,
    label_columns=[TARGET_TICKER])

#tf.keras.mixed_precision.set_global_policy('mixed_float16')

left_side_bias=tf.keras.initializers.RandomUniform(minval=-INITIALIZATION_WIDTH, maxval=0)
right_side_bias=tf.keras.initializers.RandomUniform(minval=0, maxval=INITIALIZATION_WIDTH)

#left_side_bias = right_side_bias = tf.keras.initializers.GlorotNormal()

main_input = tf.keras.Input(shape=[CONV_WIDTH,num_features])
# Level predictor permits the model to use local information to account for like, inflation,... simply

#noise_added = tf.keras.layers.GaussianNoise(stddev = 0.066667)(main_input)

level_predictor = tf.keras.layers.Dense(units=32, activation='relu',bias_initializer=right_side_bias)(main_input)
level_predictor = tf.keras.layers.Dense(units=16, activation='relu',bias_initializer=right_side_bias)(level_predictor)
level_predictor = tf.keras.layers.Dense(units=4, activation='relu',bias_initializer=right_side_bias)(level_predictor)
level_predictor = tf.keras.layers.Dense(units=1, activation='relu',bias_initializer=right_side_bias)(level_predictor)

x = tf.keras.layers.RandomZoom(height_factor=(-0.9,0.9),
                               width_factor=0,
                               fill_mode="nearest",
                               interpolation="bilinear")(main_input)
#x = tf.keras.layers.RandomContrast((0.25,1.75))(x)
x = tf.keras.layers.Conv1D(filters=52,
               kernel_size=(7,),
               activation='linear',
               bias_initializer=left_side_bias)(x)
x = tf.keras.layers.Conv1D(filters=683,
               kernel_size=(5,),
               activation='relu',
               bias_initializer=left_side_bias)(x)
x = tf.keras.layers.Conv1D(filters=1024,
               kernel_size=(4,),
               activation='relu',
               bias_initializer=left_side_bias)(x)
x = tf.keras.layers.Conv1D(filters=2048,
               kernel_size=(3,),
               activation='relu',
               bias_initializer=left_side_bias)(x)
x = tf.keras.layers.Dense(units=256, activation='relu', bias_initializer=left_side_bias)(x)
x = tf.keras.layers.Dense(units=128, activation='relu', bias_initializer=left_side_bias)(x)
x = tf.keras.layers.Dense(units=64, activation='relu', bias_initializer=left_side_bias)(x)
x = tf.keras.layers.Dense(units=1, activation='linear', bias_initializer=left_side_bias)(x)

#level_predictor = tf.keras.layers.Cropping1D(cropping=(10,1))(level_predictor)
level_predictor = tf.keras.layers.AveragePooling1D(pool_size=level_predictor.shape[1])(level_predictor)
main_output = tf.keras.layers.subtract([level_predictor, x])

conv_model = tf.keras.Model(main_input, main_output)

# Clean things out
gc.collect()

debug_dump("training CNN model")
history = compile_and_fit(conv_model, conv_window)

#conv_model.save(("base-dir/" + TARGET_TICKER + time.strftime("%d-%m-%y") + ".model"))
conv_model.save(("base-dir/" + TARGET_TICKER + "01-18-22" + ".model"))
val_performance['Conv'] = conv_model.evaluate(conv_window.val)
performance['Conv'] = conv_model.evaluate(conv_window.test, verbose=0)

#pick_file = io.open(("base-dir/" + TARGET_TICKER + time.strftime("%d-%m-%y") + ".train-perf"), "wb")
pick_file = io.open(("base-dir/" + TARGET_TICKER + "01-18-22" + ".train-perf"), "wb")

train_perf = {'Train loss': history.history['loss'][-1], 'Train mabse': history.history['mean_absolute_error'][-1]}
pickle.dump((train_perf, val_performance, performance), pick_file)
#pick_file = io.open(("base-dir/" + TARGET_TICKER + time.strftime("%d-%m-%y") + ".train-hist"), "wb")
pick_file = io.open(("base-dir/" + TARGET_TICKER + "01-18-22" + ".train-hist"), "wb")

pickle.dump(history.history, pick_file)

print("[total aggregate error, average error per guess (USD)]")
for s in performance:
       print(s, performance[s], val_performance[s])        

# ticker_renamed = TARGET_TICKER.split('Close_')[1]

# builder = (
   # BacktesterBuilder()
   # .name('{} CNN Follower'.format(ticker_renamed))
   # .balance(10_000)
   # .calendar('NYSE')
   # .compare([ticker_renamed]) # strategies to compare with
##  .live_progress() # show a progress bar
##  .live_plot(metric='Total Return (%)', min_y=None)  # we assume we are running this in a Jupyter Notebook
# )

# bt = builder.build() # build the backtest

# day_delay = 0
# last_day_pred = -1
# yesterday_price = -1

##Parts per dollar uncertainty of our uncertainty we'll tolerate holding for downturns
# holding_factor = 5

##performance_uncertainty = (((performance[s][1] + val_performance[s][1]) / 2) / 2)
# prediction_accumulator = []
# volatility_accumulator = []
# prediction_volatility_accumulator = []
# first_derivative_accumulator = []
# second_derivative_accumulator = []
##how much local price volatility effects our movement decision-making
##volatility is viewed as a risk, and makes us more likely to dump our holding
# volatility_factor = 0.15
# days_holding = 1

##logic_name = "simp"
##logic_name = "simp plus"
##logic_name = "threshold heuristic"
##logic_name = "moving heuristic"
##logic_name = "literally psychic"
# logic_name = "pretty fit applicable"

# allow_possible_leak = False
##start_date = '2021-03-01'
##end_date = '2010-09-12'
# start_date = val_df[TARGET_TICKER].index[0]
##start_date = test_df[TARGET_TICKER].index[0]
# end_date = test_df[TARGET_TICKER].index[-2]

##Important! Do this, sometimes!
# bt.prices.clear_cache()

# debug_dump(("performing backtesting with: " + logic_name))
# for day, event, b in bt[start_date:end_date]:
    # if allow_possible_leak:
        # array_from_day = np.array(df[df.index.searchsorted(day)-CONV_WIDTH:df.index.searchsorted(day)], np.float32)
    # else:
        # array_from_day = np.array(df[df.index.searchsorted(day)-CONV_WIDTH-1:df.index.searchsorted(day)-1], np.float32)

    ##HOLY FUCK -- changing this from .mean() to [-1] VASTLY improved performance, even without the leak -- WTF?!
    # cnn_prediction = conv_model.predict(tf.expand_dims(array_from_day, axis=0))[-1][-1]

    # prediction_accumulator += [cnn_prediction]
    ##Each trading day
    # if event == 'open':
        # if last_day_pred != -1:
            # time_horizon = 6
            # std_dev_predicted = 0.9
            # risk_aversity = -0.9
            ##Remember a record of price
            # volatility_accumulator += [b.price(ticker_renamed) - yesterday_price]
            ##Remember a record of our own predictions for inter-day price changes
            # prediction_volatility_accumulator += [cnn_prediction - last_day_pred]
            ##Determine how volatile the ticker is, recently
            # vol_metric = np.array(volatility_accumulator[-time_horizon-1:-1], np.float32).std()
            # bought_here = False
            # sold_here = False
            ##b.add_metric("prediction prime", np.array(prediction_volatility_accumulator[-time_horizon:]).mean())

            # if logic_name == "moving heuristic":
                # b.add_metric("prediction", cnn_prediction - last_day_pred)
                ##b.add_metric("prediction prime", np.array(prediction_volatility_accumulator[-time_horizon:]).mean())
                ##b.add_metric("reality", volatility_accumulator[-1])
                # b.add_metric("pred thresh", vol_metric*std_dev_predicted)
                # b.add_metric("sell thresh", -risk_aversity*vol_metric*std_dev_predicted*days_holding )

                ##risk-averse interday
                # if not b.portfolio.long and (cnn_prediction - last_day_pred) > vol_metric*std_dev_predicted:
                    # days_holding = 1
                    # b.long(ticker_renamed, percent_available=1)
                    # bought_here = True
                # elif b.portfolio.long and np.array(prediction_volatility_accumulator[-time_horizon:]).mean() < (-risk_aversity*vol_metric*std_dev_predicted)*days_holding:
                    # b.portfolio[ticker_renamed].long.liquidate() 
                    # sold_here = True
                # elif b.portfolio.long:
                    # days_holding += 1
            # elif logic_name == "threshold heuristic":
                # if len(prediction_volatility_accumulator) > time_horizon:
                    # grad_thresh = np.array(volatility_accumulator).mean() + np.array(volatility_accumulator).std() * 1.3
                    # total_thresh = 0
                    # pred_series = np.array(prediction_volatility_accumulator[-time_horizon:], np.float32)
                    # b.add_metric("prediction movement", np.gradient(pred_series.flatten())[-1])
                    # b.add_metric("prediction mean", pred_series.mean())
                    # b.add_metric("movement threshold", grad_thresh)
                    # if not b.portfolio.long and abs(np.gradient(pred_series.flatten())[-1]) < grad_thresh and pred_series.mean() < total_thresh:
                        # days_holding = 1
                        # b.long(ticker_renamed, percent_available=1)
                        # bought_here = True
                    # elif b.portfolio.long and abs(np.gradient(pred_series.flatten())[-1]) < grad_thresh and pred_series.mean() > -total_thresh:
                        # b.portfolio[ticker_renamed].long.liquidate() 
                        # sold_here = True
                        # days_holding = 0
                    # elif b.portfolio.long:
                        # days_holding += 1
            # elif logic_name == "simp":
                    # if not b.portfolio.long and cnn_prediction > last_day_pred:
                        # days_holding = 1
                        # b.long(ticker_renamed, percent_available=1)
                        # bought_here = True
                    # elif b.portfolio.long and cnn_prediction < last_day_pred:
                        # b.portfolio[ticker_renamed].long.liquidate() 
                        # sold_here = True
                        # days_holding = 0
            # elif logic_name == "simp plus":
                    # do_better_std = np.array(prediction_volatility_accumulator[-time_horizon:], np.float32).std()
                    # do_better_avg = np.array(prediction_volatility_accumulator[-time_horizon:], np.float32).mean()
                    # do_better = do_better_avg + do_better_std * 0.8
                    # buy_side_bias = 0.1
                    # if not b.portfolio.long and cnn_prediction > last_day_pred + do_better:
                        # days_holding = 1
                        # b.long(ticker_renamed, percent_available=1)
                        # bought_here = True
                    # elif b.portfolio.long and cnn_prediction < last_day_pred + do_better * buy_side_bias:
                        # b.portfolio[ticker_renamed].long.liquidate() 
                        # sold_here = True
                        # days_holding = 0
            # elif logic_name == "literally psychic":
                    # if not b.portfolio.long and df[TARGET_TICKER][df.index.searchsorted(day)+1] > b.price(ticker_renamed):
                        # b.long(ticker_renamed, percent_available=1)
                        # bought_here = True
                    # elif b.portfolio.long and df[TARGET_TICKER][df.index.searchsorted(day)+1] < b.price(ticker_renamed):
                        # b.portfolio[ticker_renamed].long.liquidate()
            # elif logic_name == "pretty fit applicable":
                    # if not b.portfolio.long and cnn_prediction > b.price(ticker_renamed):
                        # bought_here = True
                        # b.long(ticker_renamed, percent_available=1)
                    # elif b.portfolio.long and cnn_prediction < b.price(ticker_renamed):
                        # sold_here = True
                        # b.portfolio[ticker_renamed].long.liquidate()

            # if bought_here:
                # b.add_metric("bought here", 4.0625)
            # else:
                # b.add_metric("bought here", 0)

            # if sold_here:
                # b.add_metric("sold here", -4.0625)
            # else:
                # b.add_metric("sold here", 0)


        # last_day_pred = cnn_prediction
        # yesterday_price = b.price(ticker_renamed)
        
    # pd.to_pickle(bt.summary, ("base-dir/" + TARGET_TICKER + time.strftime("%d-%m-%y") + ".bt-stats"))
    
    # pick_file = io.open(("base-dir/" + TARGET_TICKER + time.strftime("%d-%m-%y") + ".bt-trades"), "wb")
    # pickle.dump(bt.trades, pick_file)

