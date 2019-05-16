# Applying Convolutional Auto-Encoder in Trading

In this project, I try to build a model to extract useful patterns in financial timeseries for predicting the directions of future price movements. Traditional multi-variate timeseries models (even some [modern approach like LSTM](https://www.researchgate.net/publication/327967988_Predicting_Stock_Prices_Using_LSTM)) tend to look at and extract information from each input features independently, which ignores potential correlations between inpputs. For example, looing at historical volume and adjust close prices jointly could povide new information. As such, people have been exploring using [CNN to learn spatial patterns](http://cs231n.stanford.edu/reports/2015/pdfs/ashwin_final_paper.pdf).

It is well-known that the information/noise ration is low in general for financial time-series. Here we try a novel approach, called Convolutional Auto-Encoder (CAE), which proved [successful in computer visions](https://xifengguo.github.io/papers/ICONIP17-DCEC.pdf). 

This repo contains a set of data points from the commodity-trading market. It consists 3 years, 5-mins open, high, low, close, volume and open interests. We use two years data as training-validation-test sets to build our model, and the last year data to backtest our strategy.

Our CAE and other utils are contained in 'kernel' folder, and there is a demo.ipynb for demostrating the experiment results.
