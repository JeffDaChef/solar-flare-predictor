# Forecasting major solar flares, and trying hard not to fool myself

This is the longer technical writeup. The short plain-language version is in
EXPLAINED.md. Here I go into the methods, the numbers, and the parts where I had to be
careful, because the careful parts are most of the project.

## The problem

A major solar flare is an M or X class flare. They can disrupt radio, GPS, and
satellites, so forecasting them matters. The task I set is binary. Given the recent
state of an active region on the Sun, will it produce an M or X class flare in the next
24 hours, yes or no. I train on a public benchmark called SWAN-SF and then run the same
idea live on the current Sun.

The single fact that shapes everything is the class imbalance. Major flares are roughly
60 times rarer than quiet periods. A model that always answers "no flare" is therefore
right about 98 percent of the time and is completely worthless. So accuracy is a trap,
and avoiding that trap drove most of my design decisions.

## The data

SWAN-SF (Angryk et al., 2020) is on Harvard Dataverse under DOI 10.7910/DVN/EBCFKM. It
is about 6.5 GB, delivered as five files, one per time period that the authors call a
partition. Inside each partition the instances are sorted into two folders, FL for
major flares and NF for everything else. Each instance is a 12 hour window of an active
region sampled every 12 minutes, so about 60 timesteps, with 24 photospheric
magnetic-field parameters per timestep (total unsigned flux, current helicity, and so
on). The instance is labeled by the strongest flare in the 24 hour window that follows.
Across the five partitions there are about 331,000 instances, and depending on the
partition the major-flare rate runs from 1.3 to 3.3 percent.

I deliberately used the raw version and did my own cleaning and splitting. A pre-cleaned
copy exists, but using it would hide exactly the decisions that make this honest.

## Metrics

I score with the True Skill Statistic (TSS) and the Heidke Skill Score (HSS), never raw
accuracy. From the confusion matrix,

    TSS = TP / (TP + FN) minus FP / (FP + TN)

which is sensitivity plus specificity minus one. It is 0 for any constant guesser and 1
for a perfect one, and it does not change if you alter the ratio of positives to
negatives, which is the property that makes it trustworthy under heavy imbalance.

HSS is the agreement over what chance would give, and unlike TSS it does depend on the
base rate, so the two together tell a fuller story. A high TSS with a low HSS means the
model is catching flares by raising a flood of false alarms.

There is a clean identity I lean on. On a balanced 50/50 test set, accuracy equals
(TSS + 1) / 2, so a TSS of 0.5 is the same as 75 percent balanced accuracy. I use that
only to translate scores into something intuitive, never as the operational metric.

I unit tested the metric functions against small confusion matrices worked out by hand
before building any model, so that the ruler was correct before I measured anything.

## The leakage trap

This is the most important methodological point. SWAN-SF is built with a sliding window
that steps forward an hour at a time, so two neighboring instances share about 11 of
their 12 hours. They are near-duplicates. If you split the pooled data at random, those
near-duplicates land on both sides, the model effectively sees the test answers during
training, and the reported skill is a lie. A lot of the high numbers in the literature
are partly this.

I demonstrated it in my own pipeline rather than just asserting it. With a proper split
by time partition, a random forest scores TSS 0.81 and HSS 0.26 on held-out data. With
a careless random split of the pooled instances, the same model scores TSS 0.98 and HSS
0.74. The gap is pure leakage. A second detail fell out of this: a linear model barely
moved between the two splits, because it cannot memorize individual near-duplicates,
while the high-capacity model inflated a lot. So leakage gets more dangerous exactly as
the model gets stronger, which is a useful warning for the neural networks later.

Every result in this project uses the partition split, train on early partitions and
test on completely separate later ones.

## Preprocessing

For the classical models and the multilayer net, I reduce each 12 hour window to summary
features per parameter: the mean, standard deviation, minimum, maximum, last value, and
linear trend. That is 144 numbers per instance, and the summaries skip over missing
values naturally. For the LSTM I keep the full sequence instead.

Two careful points. Missing values are imputed, not dropped, because dropping them
would throw away usable data and bias the result. And the normalization statistics are
fit on the training data only and then applied to the test data. Fitting them across
both sets is a quiet form of leakage, so I do not.

## Models

I trained four:

- Logistic regression with balanced class weights. TSS 0.833.
- Random forest, 200 trees. TSS 0.807.
- A multilayer network written from scratch in numpy. TSS 0.827.
- An LSTM written from scratch in numpy. TSS 0.829.

The numbers are not the headline. The agreement is. Four very different model families
all land between 0.81 and 0.83 on the honest split, which says the signal available in
these features is capped and no amount of model complexity breaks through. That matches
the literature, where honest reproductions sit well below the leakage-inflated headline
scores.

The neural networks are the real engineering. I wrote the forward and backward passes
by hand, including backpropagation through time for the LSTM, where the gradient has to
flow back through all 60 timesteps. To trust that, I verified it two independent ways.
First, a numerical gradient check, comparing the analytic gradients to finite-difference
estimates, which agree to about one part in a million. Second, I rebuilt the identical
networks in PyTorch, gave both the same weights and inputs, and compared. The gradients
match to roughly 1e-16, machine precision. So the from-scratch implementation is not
approximately right, it is identical to a trusted reference.

## Handling the imbalance

I used balanced class weighting in training so the model could not collapse to always
saying no. The decision threshold that turns a probability into a yes or no is chosen on
a separate validation partition by maximizing TSS, never on the test set and never on
the training set, since a model that overfits its training set would pick a threshold
that does not transfer.

I also tested whether adding each region's recent flare history helps, since that is
information the human forecasters use. First I checked those columns were not secretly
the label, and they are not, the future flare is not written into the history. Adding
history left TSS unchanged at about 0.83 but improved HSS from 0.20 to 0.25, meaning
fewer false alarms. A modest, honest gain that again does not break the ceiling.

## Calibration and the whole-Sun forecast

The model rates one region at a time, but the operational question is whether any region
on the Sun flares in the next 24 hours. I combine the per-region probabilities into a
single full-disk probability. For that combined number to be meaningful, the per-region
probabilities have to be calibrated, so I map the model's raw scores onto real-world
frequencies using a held-out partition. After calibration the model predicts 1.3 percent
on average, which matches the true rate, gives real flares an average score of 33 percent
against 1 percent for quiet regions, and is right about half the time when it calls a
region at least 30 percent likely, against a 1.3 percent base rate.

Evaluated as a daily whole-Sun forecast on a held-out year, the combined forecast
reaches AUC 0.94 and TSS 0.75, and its average matches the 8 percent daily flare rate.
That is the headline skill result.

## The operational system

The live loop pulls current active-region parameters from the NASA and Stanford JSOC
archive through the drms package (no account needed for the scalar parameters), runs the
model, and writes a dated forecast. Later it pulls GOES X-ray flux from NOAA to record
whether a major flare actually occurred, where M class is a peak flux at or above
1e-5 W/m2 and X class at or above 1e-4. It also scrapes NOAA's own daily forecast so the
scoreboard can place my forecast next to theirs, scored on the same days. The whole loop
runs once a day on its own through a free cloud scheduler.

Building this surfaced several real-world messes that no synthetic test would have
caught: NOAA occasionally reports non-physical negative flux, the JSOC server sometimes
returns text instead of a number, and a freshly emerged region with only a few timesteps
produced garbage summary statistics that fooled the model into a 100 percent forecast.
Each one is handled, the last by refusing to forecast regions that do not yet have enough
data and by never publishing a literal 0 or 100 percent.

## Honest limitations

On live data my forecast currently runs more conservative than NOAA. The reason is not a
bug. The current regions genuinely look quiet by the magnetic measurements the model
reads, while NOAA's forecasters also use region history and complexity that a single
magnetic snapshot does not contain. The public scoreboard is designed to measure this
gap over time rather than paper over it. The training data also ends in 2018 and uses the
definitive data product, while the live feed is the quick-look version, so some drift
between the two is expected.

## Reproducing it

Create a virtual environment, install requirements.txt and requirements-dev.txt, and run
pytest. The metric, loader, preprocessing, model, and validation tests all pass. The
trained model is saved under models/, so a live forecast can be issued with
PYTHONPATH=src python -m daily without downloading the training data.
