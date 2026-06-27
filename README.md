# Solar Flare Predictor

So this is a system that tries to predict big solar flares. Every day it grabs the
current state of the Sun's active regions, gives a probability that a major (M or X
class) flare goes off in the next 24 hours, saves that forecast, and then a day later
checks itself against what actually happened and against what NOAA predicted. There is
a bunch of machine learning behind it, trained on a public dataset, and I wrote the
neural networks from scratch instead of just importing them.

I mostly built this because I wanted to actually understand how it works, so the thing
I cared about the whole time was not fooling myself. Major flares are rare, and that
makes it really easy to build a model that looks amazing and is secretly useless.
Honestly most of the work was just being careful about how I measured stuff.

## How well does it work

On a full held-out year of old data, the whole-Sun forecast separates flare days from
quiet days pretty well, AUC 0.94 and TSS 0.75. TSS is the score people use for this
where 0 is no skill and 1 is perfect. It is also calibrated, so when it says 10 percent
it is actually right about 10 percent of the time.

I trained four models, everything from a one line logistic regression up to a hand
built LSTM, and they all ended up around TSS 0.83. That is kind of the real result.
This problem has a low ceiling that basically nobody beats, so a fancier model does not
magically do better, and if you ever see a number way above that range it is almost
always a data leak and not a breakthrough.

The two neural nets (a small regular one and an LSTM) are written from scratch in numpy,
backprop math and all, and I checked them against PyTorch. The gradients match to about
1e-16, which is around as exact as a computer can get.

## The honest parts, which were actually the hard part

- The metric. Flares are about 60 to 1 rare, so a model that always says "no flare"
  gets 98 percent accuracy and is completely useless. So I score with TSS and HSS
  instead of accuracy.
- The leakage trap. The data has windows that overlap a lot, so if you split it at
  random you get near copies in both training and testing and the score turns into a
  lie. I proved this on myself. A random forest jumps from a real 0.81 to a fake 0.98
  just from a sloppy random split. So I split by time period instead, everywhere.
- Live and graded in public. It runs every day on its own and scores itself against
  reality and against NOAA. Right now my model is more cautious than NOAA, because it
  only reads the magnetic field while they also know each region's recent flare
  history. I would rather show that gap than hide it.

## Running it

You need Python 3. From the project folder:

    python3 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt
    .venv/bin/python -m pip install -r requirements-dev.txt
    .venv/bin/python -m pytest

To make a live forecast (it pulls the real Sun, needs internet, no account needed):

    PYTHONPATH=src .venv/bin/python -m daily

The training data (SWAN-SF, about 6 GB) is not in the repo because it is huge. The
trained model is tiny and already saved in models/, so the daily forecast still works
without it.

## What is where

- src/ is the code, loading, metrics, cleaning, the models, and the live system.
- src/nn/ is the from scratch neural nets and the checks that prove they work.
- src/live/ is the daily forecast, the NOAA grader, and the scoreboard.
- tests/ is the tests.
- results/ has the forecast log, the scoreboard, and the evaluation numbers.
- EXPLAINED.md walks through the whole thing in plain words, step by step.
- docs/paper.md is the longer, more technical version.

## Where the data comes from

- Historical training data, the SWAN-SF benchmark on Harvard Dataverse, DOI
  10.7910/DVN/EBCFKM.
- Live active region data, NASA and Stanford's JSOC, through the drms package.
- Flare outcomes and the NOAA baseline forecast, the NOAA Space Weather Prediction
  Center.
