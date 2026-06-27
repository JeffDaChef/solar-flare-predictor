# Solar Flare Predictor

So this is basically a thing that tries to predict big solar flares before they
happen. Every day it grabs the current state of the Sun's active regions, gives a
percent chance that a major flare (the M or X class ones) goes off in the next 24
hours, saves that guess, and then a day later checks itself against what actually
happened and against what NOAA predicted. There is a bunch of machine learning behind
it that I trained on a public dataset, and I wrote the neural networks myself from
scratch instead of just importing them.

Honestly I mostly built this because I wanted to actually understand how it works.
Major flares are super rare, which makes it really easy to build a model that looks
amazing and is secretly useless, so a lot of the work was just me being careful about
how I measured stuff so I would not end up fooling myself.

## How good is it

On a full year of old data the model never saw, the whole-Sun forecast tells flare
days apart from quiet days pretty well, an AUC of 0.94 and a TSS of 0.75. TSS is the
score people use for this where 0 is useless and 1 is perfect. It is also calibrated,
so when it says like 10 percent it actually happens about 10 percent of the time.

I trained four different models, everything from a one line logistic regression up to
an LSTM I built by hand, and they all kinda landed around TSS 0.83. That is honestly
the interesting part. This problem has a low ceiling that basically nobody gets past,
so a fancier model does not magically do better, and if you ever see a score way above
that you should probably just assume something leaked.

The two neural nets (a small regular one and an LSTM) are written from scratch in
numpy, including the backprop math, and I checked them against PyTorch to make sure I
did not mess it up. They match PyTorch's gradients to about 1e-16, which is pretty
much as exact as a computer gets.

## The honest stuff, which was actually the hard part

- The score. Flares are like 60 to 1 rare, so a model that just always says "no
  flare" gets 98 percent accuracy and is completely useless. So I use TSS and HSS
  and I do not touch accuracy.
- The leakage trap. The data has windows that overlap a ton, so if you split it
  randomly you end up with near identical copies in both training and testing and
  your score goes fake. I actually showed this on my own model, a random forest jumps
  from a real 0.81 up to a fake 0.98 just from a lazy random split. So I split
  everything by time period.
- It runs live and grades itself in public. Every day it forecasts on its own and
  then scores itself against reality and against NOAA. Right now my model is more
  cautious than NOAA since it only looks at the magnetic field while they also use
  each region's recent flare history, and honestly I would rather just show that in
  the open.

## Running it

You need Python 3. From the project folder:

    python3 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt
    .venv/bin/python -m pip install -r requirements-dev.txt
    .venv/bin/python -m pytest

To make a live forecast (it pulls the real Sun, needs internet, no account):

    PYTHONPATH=src .venv/bin/python -m daily

The training data (SWAN-SF, about 6 GB) is not in the repo since it is huge. The
trained model is tiny and already saved in models/, so the daily forecast still works
without it.

## What is where

- src/ is the code, loading, metrics, cleaning, the models, and the live system.
- src/nn/ is the from scratch neural nets and the checks that prove they work.
- src/live/ is the daily forecast, the NOAA grader, and the scoreboard.
- tests/ is the tests.
- results/ has the forecast log, the scoreboard, and the evaluation numbers.
- EXPLAINED.md is me walking through the whole thing in plain words, step by step.
- docs/paper.md is the longer, more technical version.

## Where the data comes from

- The historical training data is the SWAN-SF benchmark on Harvard Dataverse, DOI
  10.7910/DVN/EBCFKM.
- The live active region data is from NASA and Stanford's JSOC, through the drms
  package.
- The flare outcomes and the NOAA baseline forecast are from the NOAA Space Weather
  Prediction Center.

