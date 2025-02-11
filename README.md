# TR-1007

This is a small architecture and training pipeline using a CNN and integrating Random Fourier Features for making predictions of the stock market.

## ElectricLake
Maintain a Pandas pickle over Stock market data samples from Polygon. Daily.

Maybe there was originally an effort to cache this into an Elastic instance?

## TR-1007-Army
Train a set of models over a set of ticker symbols; each model specializes in predicting price over the specified symbols.

Thin wrapper over TR-1007-PFC

## PFC-TR-1007
Trains a model and writes it to disk.

Augments data with Random-Fourier Features of specified geometry using a ThreadPool to accelerate performance.

Contains commented-out functionality for performing backtesting using a naive strategy, to be run after performing batch training.
