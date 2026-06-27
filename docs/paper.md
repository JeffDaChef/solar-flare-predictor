# Forecasting major solar flares, and trying really hard not to fool myself

This is the longer, more technical version. The plain walkthrough is in EXPLAINED.md.
Here I get into the methods and the numbers and the spots where I had to be careful,
because honestly the careful spots are most of the project.

## The problem

A major solar flare is an M or X class flare. They can mess with radio, GPS, and
satellites, so predicting them is worth doing. The task I set is just yes or no. Given
the recent state of an active region on the Sun, will it set off an M or X class flare
in the next 24 hours. I train on a public benchmark called SWAN-SF and then run the
same idea live on the actual current Sun.

The one fact that drives everything is how rare these are. Major flares are roughly 60
times rarer than quiet periods. So a model that just always answers "no flare" is right
about 98 percent of the time and is totally worthless. That means accuracy is a trap,
and dodging that trap shaped almost every decision I made.

## The data

SWAN-SF (Angryk et al., 2020) lives on Harvard Dataverse under DOI 10.7910/DVN/EBCFKM.
It is about 6.5 GB, split into five files, one per time period that the authors call a
partition. Inside each partition the examples are sorted into two folders, FL for major
flares and NF for everything else. Each example is a 12 hour window of an active region
sampled every 12 minutes, so about 60 timesteps, with 24 magnetic field numbers per
timestep (total unsigned flux, current helicity, that kind of thing). The example is
labeled by the strongest flare in the 24 hours that come after. Across the five
partitions there are around 331,000 examples, and depending on the partition the major
flare rate runs from 1.3 to 3.3 percent.

I used the raw version on purpose and did my own cleaning and splitting. There is a
pre-cleaned copy floating around, but using it would have hidden exactly the decisions
that make this honest, so I left it alone.

## Metrics

I score with the True Skill Statistic (TSS) and the Heidke Skill Score (HSS) instead of
plain accuracy. From the confusion matrix,

    TSS = TP / (TP + FN) minus FP / (FP + TN)

which is just sensitivity plus specificity minus one. It is 0 for any constant guesser
and 1 for a perfect one, and it does not change if you mess with the ratio of positives
to negatives, which is the property that makes it trustworthy when the data is this
lopsided.

HSS is the agreement compared to what chance would give you, and unlike TSS it does
depend on the base rate, so having both tells a fuller story. A high TSS with a low HSS
means the model is catching flares by setting off a ton of false alarms.

There is a clean little identity I lean on. On a balanced 50/50 test set, accuracy
equals (TSS + 1) / 2, so a TSS of 0.5 is the same as 75 percent balanced accuracy. I
only use that to translate scores into something my brain can picture, never as the
real metric.

I unit tested the metric functions against tiny confusion matrices I worked out by hand
before building any model, so the ruler was correct before I measured anything with it.

## The leakage trap

This is the most important point in the whole project. SWAN-SF is built with a sliding
window that steps forward an hour at a time, so two examples next to each other share
about 11 of their 12 hours. They are basically near duplicates. If you pool everything
and split it at random, those near duplicates land on both sides, the model basically
sees the test answers while training, and the skill it reports is a lie. A lot of the
big numbers in the literature are partly this.

I did not just say that, I showed it in my own pipeline. With a proper split by time
partition, a random forest gets TSS 0.81 and HSS 0.26 on held-out data. With a sloppy
random split of the pooled data, the same model gets TSS 0.98 and HSS 0.74. That whole
gap is leakage. Something else fell out of it too. A linear model barely moved between
the two splits because it cannot memorize individual near duplicates, while the strong
model ballooned. So the leakage gets more dangerous exactly as the model gets stronger,
which was a useful warning to keep in mind once the neural nets showed up.

Every result in here uses the partition split. Train on early partitions, test on
totally separate later ones.

## Cleaning the data

For the classical models and the regular neural net I squash each 12 hour window down to
summary numbers per parameter, the mean, the spread, the min, the max, the last value,
and the trend. That is 144 numbers per example, and the summaries just skip over missing
values on their own. For the LSTM I keep the full sequence instead.

Two careful things. Missing values get filled in instead of dropped, because dropping them
throws away usable data and biases the result. And the normalization stats are figured
out on the training data only and then applied to the test data. Working them out across
both sets is a sneaky little form of leakage, so I do not.

## Models

I trained four of them:

- Logistic regression with balanced class weights. TSS 0.833.
- Random forest, 200 trees. TSS 0.807.
- A regular neural net written from scratch in numpy. TSS 0.827.
- An LSTM written from scratch in numpy. TSS 0.829.

Honestly the agreement is the interesting part, more than any single number. Four
pretty different kinds of model all land between 0.81 and 0.83 on the honest split, which tells me the signal
in these features is capped and piling on model complexity does not break through. That
lines up with the literature, where honest reproductions sit well below the leakage
inflated headline scores.

The neural nets are the part I am proudest of engineering wise. I wrote the forward and
backward passes by hand, including backprop through time for the LSTM, where the
gradient has to travel back through all 60 timesteps. To actually trust that, I checked
it two separate ways. First a numerical gradient check, comparing my analytic gradients
to finite difference estimates, which agree to about one part in a million. Second I
rebuilt the same networks in PyTorch, fed both the same weights and inputs, and
compared. The gradients match to roughly 1e-16, machine precision. So my from scratch
version lines up with a trusted library down to the last decimal.

## Dealing with the imbalance

I used balanced class weighting while training so the model could not just give up and
say no to everything. The threshold that turns a probability into a yes or no I pick on
a separate validation partition by maximizing TSS, never on the test set and never on
the training set, since a model that overfits its training data would pick a threshold
that does not carry over.

I also tested whether adding each region's recent flare history helps, since that is
info the human forecasters use. First I checked those columns were not secretly the
label, and they are not, the future flare is not written into the history. Adding
history left TSS unchanged at about 0.83 but pushed HSS from 0.20 up to 0.25, so fewer
false alarms. A small honest gain that again does not break the ceiling.

## Calibration and the whole-Sun forecast

The model rates one region at a time, but the real daily question is whether any region
on the Sun flares in the next 24 hours, so I combine the per region probabilities into
one full-disk number. For that combined number to mean anything the per region
probabilities have to be calibrated, so I map the model's raw scores onto real world
frequencies using a held-out partition. After calibrating, the model predicts 1.3
percent on average, which matches the true rate, gives real flares an average score of
33 percent against 1 percent for quiet regions, and is right about half the time when
it calls a region at least 30 percent likely, against a 1.3 percent base rate.

Looked at as a daily whole-Sun forecast on a held-out year, the combined forecast hits
AUC 0.94 and TSS 0.75, and its average matches the 8 percent daily flare rate. That is
the headline skill result.

## The live system

The live loop pulls current active region parameters from NASA and Stanford's JSOC
archive through the drms package (no account needed for the scalar parameters), runs the
model, and writes a dated forecast. Later it pulls GOES X-ray flux from NOAA to record
whether a major flare actually happened, where M class is a peak flux at or above
1e-5 W/m2 and X class at or above 1e-4. It also grabs NOAA's own daily forecast so the
scoreboard can put mine right next to theirs on the same days. The whole loop runs once
a day by itself through a free cloud scheduler.

Building this turned up a bunch of real world messes that no test on clean data would
have caught. NOAA sometimes reports a non-physical negative flux, the JSOC server
sometimes hands back text instead of a number, and one time a freshly emerged region
with only a few timesteps produced garbage summary stats that fooled the model into a
100 percent forecast. Each one is handled, the last by refusing to forecast regions that
do not have enough data yet and by never publishing a literal 0 or 100 percent.

## What it does not do well

On live data my forecast is currently more cautious than NOAA. I dug in and it is real.
The current regions genuinely look quiet by the magnetic measurements my model reads, while
NOAA's forecasters also use region history and complexity that a single magnetic
snapshot just does not have. The public scoreboard is there to measure that gap over
time instead of papering over it. The training data also stops in 2018 and uses the
final data product, while the live feed is the quick look version, so a bit of drift
between them is expected.

## Reproducing it

Make a virtual environment, install requirements.txt and requirements-dev.txt, and run
pytest. The metric, loader, cleaning, model, and validation tests all pass. The trained
model is saved in models/, so you can issue a live forecast with
PYTHONPATH=src python -m daily without downloading the training data.

