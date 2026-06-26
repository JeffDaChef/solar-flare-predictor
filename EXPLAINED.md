# What this project does, in plain words

This is my solar flare predictor. The idea is to train a model that looks at
measurements of the Sun's magnetic field and guesses whether a big flare (an M or
X class one) will happen in the next 24 hours. Big flares are rare, so honestly a
lot of this project is about not fooling myself into thinking the model is better
than it really is.

I am building it in steps, and I will add to this file as I go. The heavier
technical writeup lives in docs/paper.md so this one can stay short and readable.

## Step 1, the scoring (done first on purpose)

Before building any model I built the scoring, because if the scoring is wrong then
every result after it is wrong too. So it had to come first.

Here is the trap. Big flares happen maybe 1 day out of 60. So a model that just
says "no flare" every single time is right about 98 percent of the time, and it is
completely useless. Plain accuracy lies in a situation like this.

So accuracy is not my main score. I use something called TSS, the True Skill
Statistic. It rewards actually catching flares and it punishes false alarms, and a
lazy "always no" model scores a flat 0 on it. On TSS, 0 means no real skill and 1
means perfect. I also keep a second score called HSS as a sanity check.

To prove it to myself I made a fake Sun with 100 flare days and 6000 quiet days. A
model that always says no got 98 percent accuracy but a TSS of 0. A model that
actually caught 70 of the 100 flares scored lower on accuracy, 95 percent, but a
TSS of 0.65. The lazy model looked better on accuracy and was worse in every way
that matters. That is exactly why I score with TSS.

There is also a neat link I will use later. If you test on a balanced set with
equal flare and no flare days, then accuracy becomes meaningful again, and it lines
up with TSS by accuracy equals (TSS plus 1) divided by 2. So later I plan to report
two numbers, the honest one on the real rare data, and the easy to read one on a
balanced 50/50 set.

## Where this part lives

- src/metrics.py has the scoring functions (TSS, HSS, and a few helpers).
- tests/test_metrics.py checks them against small examples I worked out by hand, so
  I can trust the numbers. Right now all of those tests pass.

## Step 2, getting and loading the data

The data is a benchmark called SWAN-SF. It is about 6.5 gigabytes, split into 5
files, one per time period (they call them partitions). Those 5 separate periods are
the whole reason I can avoid the leakage trap later, since I can train on some
periods and test on completely different ones.

Inside each partition the examples are sorted into two folders. FL means a major
flare happened (the M and X class ones) and NF means it did not. So the folder name
is literally the answer the model is trying to predict. Each example is one 12 hour
window of the Sun recorded every 12 minutes, so about 60 measurements, and each
measurement has the 24 magnetic numbers I care about.

I wrote a loader in src/load.py that opens these files, pulls out the 24 numbers for
each example, and tags it 1 for flare or 0 for no flare based on its folder. Some
values in the files are missing or literally say None, so the loader marks those as
blanks for now and I fill them in during the cleaning step. The tests in
tests/test_load.py check that it reads the real files correctly, and they pass.

One thing I learned the hard way. When I first started the big download in the
background, the computer told me it finished fine, but when I actually looked,
nothing had downloaded. The shell had quietly messed up. Same lesson as the metrics,
do not trust the "it worked" message, go check the real thing.

## Where this part lives

- src/load.py reads the SWAN-SF files into memory.
- tests/test_load.py checks the loader against the real data.

## Step 3, cleaning the data and the first model

Before a model can learn, the raw measurements need cleaning. Each example is a 12
hour window with 24 magnetic numbers measured about 60 times, and some of those are
missing. Instead of feeding the whole messy time series in, I boil each example down
to a handful of summary numbers per measurement (its average, how much it varied, its
lowest and highest, its last value, and its trend across the window). That is 144
numbers per example, and the summaries naturally skip over the missing bits.

One careful detail. I work out the normalization (the rescaling that puts every
number on a fair footing) using only the training data, then apply it to the test
data. If I had used the test data to figure out the scaling, the model would have
quietly peeked at the answers. Small thing, but it is the line between honest and not.

Then I trained two simple models, a logistic regression and a random forest, on the
proper split (train on early years, test on later years). Both landed around TSS 0.8,
which at first looks incredible, almost like matching the best research. But the HSS
told the real story, only about 0.2 to 0.26. High TSS with low HSS means the model is
catching most flares by raising a flood of false alarms. It flagged about 16,000
windows as dangerous when only about 2,000 really were. So there is real signal here,
but the model is trigger happy, and TSS alone would have flattered it. This is exactly
why I never trust a single number.

The best part. I ran the same models a second time but split the data at random
instead of by time, which is the classic mistake that lets near-copies leak between
training and test. The linear model barely moved. But the random forest jumped from
TSS 0.81 to 0.98 and HSS 0.26 to 0.74, suddenly looking nearly perfect. That jump is
pure cheating. Seeing it happen in my own code is the clearest proof of why the
time-based split matters. Powerful models get fooled by the leak, simple ones do not,
and if I had not been careful I would have proudly reported a fake 0.98.

## Where this part lives

- src/preprocess.py cleans the data and makes the 144 summary numbers per example.
- src/baseline.py trains the first models and runs the honest-vs-leaky comparison.
- results/baseline.txt has the numbers from that run.

## Step 4, going live

The whole point of this project is that it does not just run on old data, it runs on
the Sun as it is right now. There are two live pieces.

The grader (src/live/score.py). This pulls NOAA's real X-ray measurements of the Sun
and decides whether a major flare actually happened on a given day. A flare counts as
major (M or X class) when the X-ray brightness crosses a fixed line. I pointed it at
the live feed and it correctly flagged the M-class flares on June 20 and 21. This is
the half that grades my forecasts later.

The live input (src/live/fetch.py). This pulls the current magnetic numbers for every
active region on the Sun right now, the same 24 measurements the model trained on, and
shapes each region into the same 144-number summary. When I ran it, it pulled 20
active regions off the live Sun and turned each into model-ready input.

Good news on setup. Pulling these live numbers needs no account or registration at
all. Registration is only for downloading the raw image files, which I am not doing.
Two more real-world messes showed up and got handled. NOAA sometimes reports zero or
negative brightness for bad readings (impossible, so I drop those), and the solar data
server sometimes returns the text "Invalid KeyLink" instead of a number (I treat that
as a blank, which the cleaning step already fills in).

## Where this part lives (live)

- src/live/score.py grades a day using NOAA's real flare record.
- src/live/fetch.py pulls the current Sun's magnetic numbers for every active region.

## Step 5, the first live forecast, and a lesson about lying with confidence

I wired the whole thing together. It pulls today's active regions, runs each through
the model, and combines them into one number, the chance that any major flare hits in
the next 24 hours. It writes that forecast to a dated log. The headline machine runs.

But the first number it gave was a quiet lie, and catching that was the interesting
part. My first model was simple and untuned, and it confidently said 2.2 percent. That
sounds fine until you realize a model trained on something that only happens 2 percent
of the time learns to mumble low numbers about everything, so its 2.2 percent did not
really mean anything.

So I calibrated it, which means taking the model's raw scores and mapping them onto
real-world frequencies using data it never trained on. After that the honest forecast
for today dropped to 0.2 percent. Lower, but true. And I checked the calibrated model
on a full year of held-out data to make sure it was not just broken. It is genuinely
good. On average it predicts 1.3 percent, which is exactly the real flare rate, so it
is honest. It gives real flares an average score of 33 percent against 1 percent for
quiet regions, so it really can tell them apart. And when it calls a region at least 30
percent likely to flare, it is right about half the time, against a base rate of 1.3
percent, which is a big lift.

So why is today so low? Because today's active regions genuinely look moderate to the
model, weaker than the big flare-makers it learned from. A small flare still slipped
through, which is just the honest difficulty of this problem. No model is good at it,
and the right answer to that is a long public track record, not a cherry-picked day.

## Where this part lives (the forecast)

- src/production.py trains and saves the calibrated model.
- src/live/forecast.py issues a forecast from the live Sun and logs it.
- results/forecast_log.jsonl is the running forecast log.

## Step 6, the scoreboard

A forecast is worthless if I never check it. The scoreboard reads every forecast in
the log, waits until its 24 hour window has fully passed, then asks NOAA's real record
whether a major flare actually happened in that window. From all the graded forecasts
it works out my running skill (TSS and HSS) plus a Brier score, which measures how
honest my probabilities were, not just the yes or no. Right now it correctly says
there is one forecast still pending, since its window has not closed yet. It fills in
on its own as days pass, and that growing record is the whole public point of the
project.

## Where this part lives (the scoreboard)

- src/live/scoreboard.py grades past forecasts and tracks the running scores.
- results/scoreboard.json holds the latest scoreboard.

## Step 7, the neural network, built from scratch

This is the hard part, and the whole point of it is the engineering, not the score. I
built a small neural network from scratch in numpy. That means I wrote the math that
lets it learn, the backpropagation, by hand, instead of letting a library do it for
me. The danger with doing that yourself is getting the math subtly wrong and never
finding out. So I also wrote a gradient check, which compares my hand-derived math
against a slow but foolproof numerical estimate. They agree to about one part in a
million, so I know the backprop is right.

Then I trained it on the real flare data. It scored TSS 0.827, basically tied with the
plain logistic regression (0.833) and the random forest (0.807). That is exactly what
I expected, and it is the honest story. A neural network does not magically beat the
simple models on this problem, because the ceiling is low for everyone. The reason
this net matters is not the number, it is that I built and verified the entire learning
engine myself.

Then I built the harder one, an LSTM, also from scratch. An LSTM reads the 12 hour
window step by step like a little memory, and training it means sending the gradient
backward through all 60 steps, which is the trickiest math in the project. I proved
that math correct with the same gradient check, then trained it on the flare
sequences. It scored TSS 0.829, right alongside everything else. Four different
models, from a one-line logistic regression to a hand-built LSTM, all land in the
same narrow band around 0.81 to 0.83. That agreement is itself the finding. The
problem has a hard ceiling, and no amount of model muscle breaks through it. If I
ever saw a number far above this band, my first guess would be a leak, not a
breakthrough.

To really be sure my hand-written math was right, I checked it against PyTorch, the
standard deep learning library that figures these gradients out automatically. I
built the exact same networks in PyTorch, gave both versions the same weights and the
same inputs, and compared. They matched to machine precision. The gradients differed
by about 1e-16, which is basically the smallest gap a computer can even represent. So
my from-scratch learning engine is not just close to a trusted reference, it is
identical to it, down to the last decimal. That is the strongest proof I can give that
I built it correctly and actually understand it, instead of leaning on a library.

## Where this part lives (the network)

- src/nn/layers.py, the network pieces (including the LSTM) with hand-written
  forward and backward math.
- src/nn/losses.py and src/nn/optim.py, the loss function and the optimizer.
- src/nn/gradcheck.py, the proof that the math is internally correct.
- src/nn/validate_torch.py, the check that it matches PyTorch to machine precision.
- src/nn/train_mlp.py and src/nn/train_lstm.py, train the two nets on the flares.

## Step 8, forecasting the whole Sun, and an honest reckoning with NOAA

My model rates one active region at a time. The real daily question is whether any
region on the Sun will flare in the next 24 hours, so I combine the regions into one
full-disk number. I tested that combined forecast on a full held-out year of history,
and it is genuinely good. It separates flare days from quiet days with an AUC of 0.94
(1.0 is perfect, 0.5 is a coin flip) and a TSS of 0.75, and it is well-calibrated, it
predicts flares on about 10 percent of days when the real rate is 8 percent.

Then I lined it up against NOAA, the government forecasters, on live data, and it told
me something humbling. On the first graded day our forecast was far too low and a flare
happened, and our live numbers run well under NOAA's. I dug into why, and it is not a
bug. The current regions genuinely look quiet by the magnetic measurements my model
reads, their values sit near the quiet-region baseline, while NOAA forecasts higher
because its forecasters also use each region's recent flare history and complexity,
which my magnetic snapshot does not include. So on live data my model is currently more
timid than the experts. That is an honest limitation, not a failure, and the whole
point of the public scoreboard is to show it truthfully over time instead of hiding it.

One thing I want to be honest about. I first guessed the problem was bad calibration
and was about to fix that. Instead I measured, and the measurement proved my guess
wrong, the historical forecast was already well-calibrated. Checking before fixing
saved me from solving the wrong problem.

## Where this part lives (full-disk and NOAA)

- src/fulldisk.py measures the whole-Sun daily forecast on held-out history.
- src/live/noaa.py pulls NOAA's own forecast so the scoreboard can compare.
- results/fulldisk.json holds the historical full-disk result.

## Step 9, does recent flare history help

NOAA's forecasters partly beat us because they look at each region's recent flare
history, which my magnetic snapshot ignored. The data actually includes that history,
so I tested adding it. First I checked it was not secretly the answer, and it is not,
a region that flares in the future does not have that future flare written into its
history columns. Then I trained the same model with and without it.

The result was honest and modest. Adding flare history left the TSS basically unchanged
(about 0.83 either way), but it improved the HSS from 0.20 to 0.25, which means fewer
false alarms. So history sharpens the model a little, but it does not break the skill
ceiling. Even the extra information the experts use does not crack this problem open,
which is the same lesson the four models taught me. And on its own it does not close
the live gap with NOAA, because that gap is more about the current regions genuinely
looking quiet to the magnetic measurements than about one missing feature.

## Where this part lives (history experiment)

- src/history_experiment.py runs the with-versus-without comparison.
- results/history_experiment.txt has the numbers.

## Step 10, running it every day, and a bug the test caught

To grow a real track record without me doing anything, the forecast needs to run on
its own every day. I set that up with GitHub Actions, a free scheduler that runs in the
cloud once a day, issues the forecast, updates the scoreboard, and saves the log back to
the repo. No computer of mine has to be on.

Testing that daily job immediately caught a bug, which is the best argument for testing
it. One run forecast 100 percent, which is never a real answer. I traced it to a single
active region that had just rotated into view with only 3 measurements instead of the
usual 60. With that little data its summary numbers were garbage, and the model, which
only ever saw full 12 hour windows in training, got fooled into a near-certain score.
The fix is simple and correct, skip any region that does not have enough data yet, and
never publish a literal 0 or 100 percent. After the fix the same day reads 0.5 percent,
which is honest. The established regions really do look quiet to the magnetic
measurements, even though NOAA forecasts higher using region history my model ignores.

## Where this part lives (automation)

- src/daily.py is the once-a-day job, forecast plus scoreboard.
- .github/workflows/daily.yml is the cloud scheduler.

## Where it stands now

That is the whole build. The short version of where it landed:

- Four models, from a one-line logistic regression to a hand-built LSTM, all around TSS
  0.83 on an honest time-based split. The from-scratch nets match PyTorch to machine
  precision.
- The whole-Sun daily forecast scores AUC 0.94 and TSS 0.75 on a held-out year, and it
  is calibrated.
- A live system pulls the real Sun every day, forecasts, and grades itself against both
  reality and NOAA. Right now it runs more conservative than NOAA, and the public
  scoreboard tracks that honestly over time.

The thing I am most glad about is not any single number. It is that every time a result
looked too good, I dug in and it turned out to be a leak, an artifact, or the metric
flattering me, and I fixed it instead of reporting it. The deeper technical version of
all this is in docs/paper.md.
