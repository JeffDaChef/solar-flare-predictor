# Solar Flare Predictor

This is a system that forecasts major solar flares. Every day it pulls the current
state of the Sun's active regions, estimates the chance that an M or X class flare
happens in the next 24 hours, writes that forecast down, and later grades itself
against what the Sun actually did and against NOAA's own forecast. Behind it are
machine learning models trained and honestly evaluated on a public benchmark, with
the neural networks written from scratch.

I built this mostly to understand it, so the theme running through the whole thing is
not fooling myself. Major flares are rare, which makes it easy to build a model that
looks amazing and is useless. Most of the real work went into measuring honestly.

## What it does, and how well

On a full held-out year of historical data, the whole-Sun daily forecast tells flare
days from quiet days with an AUC of 0.94 and a TSS of 0.75, and it is well-calibrated,
predicting flares on about 10 percent of days when the real rate is 8 percent. TSS is
the standard skill score for this problem, where 0 is no skill and 1 is perfect.

I trained four different models, from a one-line logistic regression to a hand-built
LSTM, and they all land in a narrow band around TSS 0.83. That agreement is the point.
This problem has a low ceiling that nobody beats by much, so a fancier model does not
win here, and a number far above the band almost always means a data leak rather than
a breakthrough.

The two neural networks (a small multilayer net and an LSTM) are written from scratch
in numpy, backpropagation included, then checked against PyTorch. They match PyTorch's
gradients to machine precision, about 1e-16.

## The honesty parts, which are the actual hard part

- The metric. At 60 to 1 odds, an always-no-flare model scores 98 percent accuracy and
  has zero real skill, so I use TSS and HSS instead of accuracy.
- The leakage trap. The data has heavily overlapping time windows, so a random split
  lets near-copies leak between training and test. I proved this in my own code. A
  random forest jumps from a real TSS of 0.81 to a fake 0.98 on a careless random
  split. The honest way is to split by the dataset's separate time periods, which is
  what I do everywhere.
- Live and graded in public. The forecast runs every day on its own and scores itself
  against reality and against NOAA, the government forecasters. Right now my model is
  more conservative than NOAA, because it only reads the magnetic field while they also
  use each region's recent flare history. I show that gap honestly instead of hiding it.

## Running it

You need Python 3. From the project folder:

    python3 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt
    .venv/bin/python -m pip install -r requirements-dev.txt
    .venv/bin/python -m pytest

To issue a live forecast (it pulls the real Sun, needs internet, no account required):

    PYTHONPATH=src .venv/bin/python -m daily

The historical training data (SWAN-SF, about 6 GB) is not in the repo. The model is
small and already trained and saved under models/, so the daily forecast runs without
the training data.

## Layout

- src/ loading, metrics, preprocessing, the models, and the live system.
- src/nn/ the from-scratch neural networks and the checks that verify them.
- src/live/ the daily forecast, the NOAA grader, and the scoreboard.
- tests/ the unit tests.
- results/ the forecast log, the scoreboard, and the evaluation numbers.
- EXPLAINED.md a plain-language walkthrough of the whole thing, step by step.
- docs/paper.md the deeper technical writeup.

## Where the data comes from

- Historical training data, the SWAN-SF benchmark on Harvard Dataverse, DOI
  10.7910/DVN/EBCFKM.
- Live active-region data, the NASA and Stanford JSOC archive, through the drms package.
- Flare outcomes and the baseline forecast, the NOAA Space Weather Prediction Center.
