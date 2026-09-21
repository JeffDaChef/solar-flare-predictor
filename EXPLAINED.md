# What this project does, in plain words

Ok so this is my solar flare predictor. The idea is to train a model that looks at
measurements of the Sun's magnetic field and guesses whether a big flare (an M or X
class one) is gonna happen in the next 24 hours. Big flares are pretty rare, so
honestly a huge part of this whole project was just me trying not to fool myself into
thinking the model is better than it actually is.

I built it in steps and kept adding to this file as I went. The heavier technical
version is in docs/paper.md so this one can stay short and readable.

## Step 1, the scoring (did this first on purpose)

Before building any model I built the scoring, because if the scoring is wrong then
every result after it is wrong too, so it kinda had to come first.

Here is the trap. Big flares happen maybe 1 day out of 60. So a model that just says
"no flare" every single time is right like 98 percent of the time and is completely
useless. Accuracy basically lies to you when the thing is this rare.

So accuracy is not my main score. I use this thing called TSS, the True Skill
Statistic. It rewards actually catching flares and it punishes false alarms, and that
lazy "always no" model gets a flat 0 on it. On TSS, 0 means no real skill and 1 means
perfect. I also keep a second score called HSS as a sanity check.

To prove it to myself I made a fake Sun with 100 flare days and 6000 quiet days. The
model that always says no got 98 percent accuracy but a TSS of 0. A model that
actually caught 70 of the 100 flares got a worse accuracy, like 95 percent, but a TSS
of 0.65. So the lazy one looked better on accuracy and was secretly garbage, which is
basically the whole reason I score with TSS.

There is also a neat little link I use later. If you test on a balanced set with
equal flare and quiet days, accuracy becomes useful again, and it lines up with TSS
by accuracy equals (TSS plus 1) divided by 2. So later I report two numbers, the
honest one on the real rare data and the easy to read one on a balanced 50/50 set.

## Where this part lives

- src/metrics.py has the scoring functions (TSS, HSS, and a few helpers).
- tests/test_metrics.py checks them against small examples I worked out by hand, so
  I can actually trust the numbers. They all pass.

## Step 2, getting and loading the data

The data is a benchmark called SWAN-SF. It is about 6.5 gigabytes, split into 5 files,
one per time period (they call them partitions). Those 5 separate periods are the whole
reason I can dodge the leakage trap later, since I can train on some periods and test
on completely different ones.

Inside each partition the examples are sorted into two folders. FL means a major flare
happened (the M and X class ones) and NF means it did not, so the folder name is
literally the answer the model is trying to guess. Each example is one 12 hour window
of the Sun recorded every 12 minutes, so about 60 measurements, and each measurement
has the 24 magnetic numbers I care about.

I wrote a loader in src/load.py that opens these files, pulls out the 24 numbers for
each example, and tags it 1 for flare or 0 for no flare based on its folder. Some
values in the files are missing or literally say None, so the loader marks those as
blank for now and I fill them in during the cleaning step.

One thing that got me here. When I first started the big download in the background,
the computer told me it finished fine, but when I actually looked, nothing had
downloaded. The shell quietly messed up. Same lesson as the metrics honestly, do not
trust the "it worked" message, go check the real thing.

## Where this part lives

- src/load.py reads the SWAN-SF files into memory.
- tests/test_load.py checks the loader against the real data.

## Step 3, cleaning the data and the first model

Before a model can learn anything, the raw measurements need cleaning. Each example is
a 12 hour window with 24 magnetic numbers measured about 60 times, and some of them are
missing. Instead of feeding the whole messy time series in, I squash each example down
to a few summary numbers per measurement (its average, how much it moved around, its
lowest and highest, its last value, and its trend across the window). That is 144
numbers per example, and the summaries just skip over the missing bits on their own.

One careful thing. I figure out the normalization (the rescaling that puts every
number on a fair footing) using only the training data, then apply it to the test
data. If I had used the test data to figure out the scaling, the model would have
basically peeked at the answers. Small thing, but it is a quiet way to cheat without
realizing it.

Then I trained two simple models, a logistic regression and a random forest, on the
proper split (train on early years, test on later years). Both landed around TSS 0.8,
which at first looks insane, almost like matching the best research. But the HSS told
the real story, only like 0.2 to 0.26. A high TSS with a low HSS means the model is
catching most flares by setting off a flood of false alarms. It flagged about 16,000
windows as dangerous when only about 2,000 actually were. So there is real signal here,
but the model is trigger happy, and TSS by itself would have flattered me. This is
basically why I never trust a single number.

The best part though. I ran the same models a second time but split the data at random
instead of by time, which is the classic mistake that lets near copies leak between
training and testing. The linear model barely moved. But the random forest jumped from
TSS 0.81 to 0.98 and HSS 0.26 to 0.74, suddenly looking almost perfect. That jump is
pure cheating. Seeing it happen in my own code is honestly the clearest proof of why
the time based split matters. If I had been sloppy I would have proudly reported a fake
0.98.

## Where this part lives

- src/preprocess.py cleans the data and makes the 144 summary numbers per example.
- src/baseline.py trains the first models and runs the honest vs leaky comparison.
- results/baseline.txt has the numbers from that run.

## Step 4, going live

The whole point of this project is that it does not just sit on old data, it runs on
the Sun as it is right now. There are two live pieces.

The grader (src/live/score.py). This pulls NOAA's real X-ray measurements of the Sun
and decides whether a major flare actually happened on a given day. A flare counts as
major (M or X class) when the X-ray brightness crosses a fixed line. I pointed it at
the live feed and it correctly flagged the M class flares on June 20 and 21. This is
the half that grades my forecasts.

The live input (src/live/fetch.py). This pulls the current magnetic numbers for every
active region on the Sun right now, the same 24 measurements the model trained on, and
shapes each region into the same 144 number summary. When I ran it, it pulled 20 active
regions off the live Sun and turned each one into model ready input.

Good news on setup. Pulling these live numbers needs no account or registration at all.
Two more real world messes showed up and got handled. NOAA sometimes reports zero or
negative brightness for bad readings (which is impossible, so I drop those), and the
solar data server sometimes hands back the text "Invalid KeyLink" instead of a number
(I treat that as blank, which the cleaning step already fills in).

A third mess showed up later, on August 16, when the solar data server just stopped
answering. Not an error, it simply never replied. My code already retried three times,
but the tries were only 15 seconds apart and each one sat there waiting on the network
for over two minutes before giving up, so the whole thing burned almost eight minutes
and still missed the day. Now each try gives up after 45 seconds, and the waits between
tries stretch out (30 seconds, then 2 minutes, then 5) so the retries cover a longer
outage instead of all landing in the same bad minute. The August 16 forecast is still
missing from the log and I left it that way on purpose. Filling it in now, when I
already know what the Sun did that day, would not be a real forecast.

## Where this part lives (live)

- src/live/score.py grades a day using NOAA's real flare record.
- src/live/fetch.py pulls the current Sun's magnetic numbers for every active region.

## Step 5, the first live forecast, and a lesson about lying with confidence

I wired the whole thing together. It pulls today's active regions, runs each one
through the model, and combines them into one number, the chance that any major flare
hits in the next 24 hours. It writes that forecast to a dated log. The headline machine
runs.

But the first number it gave was a quiet lie, and catching that was the fun part. My
first model was simple and untuned, and it confidently said 2.2 percent. That sounds
fine until you realize a model trained on something that only happens 2 percent of the
time learns to mumble low numbers about everything, so its 2.2 percent did not really
mean anything.

So I calibrated it, which basically means taking the model's raw scores and mapping
them onto real world frequencies using data it never trained on. After that the honest
forecast for that day dropped to 0.2 percent, which felt low but was the truth. And I checked
the calibrated model on a full year of held out data to make sure it was not just
broken. It is genuinely good. On average it predicts 1.3 percent, which is exactly the
real flare rate, so it is honest. It gives real flares an average score of 33 percent
against 1 percent for quiet regions, so it really can tell them apart. And when it
calls a region at least 30 percent likely to flare, it is right about half the time,
against a base rate of 1.3 percent, which is a big lift.

So why was that day so low? Because the active regions that day genuinely looked
moderate to the model, weaker than the big flare makers it learned from. A small flare
still slipped through, which is just the honest difficulty of this problem. Nobody is
good at it, and the right answer to that is a long public track record instead of a
cherry picked day.

## Where this part lives (the forecast)

- src/production.py trains and saves the calibrated model.
- src/live/forecast.py issues a forecast from the live Sun and logs it.
- results/forecast_log.jsonl is the running forecast log.

## Step 6, the scoreboard

A forecast is worthless if I never check it. The scoreboard reads every forecast in the
log, waits until its 24 hour window has fully passed, then asks NOAA's real record
whether a major flare actually happened in that window. From all the graded forecasts
it works out my running skill (TSS and HSS) plus a Brier score, which measures how
honest my probabilities were and not just the yes or no. It fills in on its own as days
pass, and that growing record is the whole public point of the project.

Two things about it took me a while to get right. NOAA's X-ray feed only goes back 7
days, and the scoreboard used to regrade every forecast from scratch on every run. So
once a day fell out of that 7 day window there was no flare data for it anymore and it
quietly got marked as no flare. It was erasing its own history, and it had wiped 12 real
flare days before I caught it. Now a day gets graded once, when its window closes and
the data is actually there, and that verdict is saved and never recomputed. If the data
does not cover the window the day just stays ungraded instead of being scored as a no.

The other thing is that my model and NOAA get called at different thresholds, mine at
10 percent and NOAA at 50 percent. That is not me picking a friendly number. My model is
calibrated so its full disk probability is an honest small number, and 10 percent is the
cutoff src/fulldisk.py learned on held out days. NOAA issue human percentages on a
totally different scale. Scoring both at one cutoff makes whichever one is on the wrong
scale look broken, which is exactly the bug I had in both directions.

Fair warning about one seam in the record. I retrained the production model on
2026-07-31 to fix the held out data problem further down, which moved the deployed
threshold from 0.10 to 0.24. The forecasts logged before that date came from the old
model, so the running scoreboard is scoring them at a cutoff that was derived for the
new one. I cannot go back and recompute them because I only ever logged the resulting
probability, not the magnetic inputs. Every forecast from now on records which model
made it. The live TSS dropped from 0.15 to 0.08 when the threshold changed, and I would
rather show that than keep the friendlier cutoff I got from the leaky setup.

The deeper reason the live numbers are weak is not the threshold anyway. The forecasts
in the log average about 2 percent, and the real flare rate in this live stretch has been
around 39 percent, so they under-forecast by roughly 17 times against what actually
happened. The Sun is a lot busier now than in the data I trained on, which ends in 2018.

Important caveat on that number. Every forecast in the log was made by the old model.
I checked the retrained one against the old one on six recent days using the exact same
magnetic inputs, and it forecasts about 3 times higher (0.21 average against 0.07). So
the 17 times gap describes the model I was running, not the one running now, and the new
one looks a good deal closer to reality. Six days is not enough to put a number on it.
The scoreboard has to earn that the slow way.

It is worth splitting that into two separate problems, because they have different
fixes. One is that the numbers come out too small, and that one is fixable by rescaling.
The other is that the model has gotten worse at ranking which days are risky, and that
one is not. The old model's AUC on live data is 0.66, against 0.94 on held out history.
Any rescaling
you can name, prior correction, refitting the calibration, moving the threshold, is a
monotone transform, so it leaves AUC exactly where it was. It makes the numbers honest
without making the model smarter.

I should be careful here though. 33 graded days is not much, and the 95 percent interval
on that 0.66 runs from 0.46 to 0.85, which includes coin flipping. So I can say the live
skill looks a lot worse than the historical skill, but I cannot say much more than that
yet. The scoreboard needs to keep running.

## Where this part lives (the scoreboard)

- src/live/scoreboard.py grades past forecasts and tracks the running scores.
- results/scoreboard.json holds the latest scoreboard.

## Step 7, the neural network, built from scratch

This is the hard part, and the whole point of it is the engineering. I built a small
neural network from scratch in numpy. That means I wrote the math that lets it learn,
the backpropagation, by hand, instead of letting a library do it for me. The danger
with doing that yourself is getting the math subtly wrong and never finding out. So I
also wrote a gradient check, which compares my hand done math against a slow but
foolproof numerical estimate. They agree to about one part in a million, so I know the
backprop is right.

Then I trained it on the real flare data. It scored TSS 0.827, which is basically tied
with the plain logistic regression (0.833) and the random forest (0.807). That is
exactly what I expected. A neural network does not magically beat the simple models on
this problem because the ceiling is low for everyone. What makes this net worth it is
that I built and verified the whole learning engine myself.

Then I built the harder one, an LSTM, also from scratch. An LSTM reads the 12 hour
window step by step like a little memory, and training it means sending the gradient
backward through all 60 steps, which is the trickiest math in the project. I proved
that math correct with the same gradient check, then trained it on the flare sequences.
It got TSS 0.829, right alongside everything else. So four pretty different models,
from a one line logistic regression to a hand built LSTM, all land in a narrow band
around 0.81 to 0.83. That agreement is honestly the finding. The problem has a hard
ceiling and no amount of model muscle gets through it. If I ever saw a number way above
that band, my first guess would be a leak.

To really be sure my hand done math was right, I checked it against PyTorch, the
standard deep learning library that figures these gradients out automatically. I built
the same networks in PyTorch, gave both versions the same weights and the same inputs,
and compared. They matched to machine precision. The gradients were off by about 1e-16,
which is basically the smallest gap a computer can even represent. So my from scratch
engine lines up with a trusted library down to the last decimal, which is about the
strongest proof I can give that I actually understand it.

## Where this part lives (the network)

- src/nn/layers.py, the network pieces (including the LSTM) with hand written forward
  and backward math.
- src/nn/losses.py and src/nn/optim.py, the loss function and the optimizer.
- src/nn/gradcheck.py, the proof that the math is internally correct.
- src/nn/validate_torch.py, the check that it matches PyTorch to machine precision.
- src/nn/train_mlp.py and src/nn/train_lstm.py, train the two nets on the flares.

## Step 8, forecasting the whole Sun, and an honest reckoning with NOAA

My model rates one region at a time, but the real daily question is whether any region
on the Sun will flare in the next 24 hours, so I combine the regions into one full disk
number. I tested that combined forecast on a full held out year, and it is genuinely
good. It separates flare days from quiet days with an AUC of 0.94 (1.0 is perfect, 0.5
is a coin flip) and a TSS of 0.71, and it is well calibrated, predicting flares on
about 9 percent of days when the real rate is 8 percent.

Getting that number honest took two fixes and I want to be clear about both, because I
had it wrong for a while. The first is which data the model is allowed to see. The
production model used to calibrate on partition 5 and then get tested on partition 5,
so the "held out year" was not actually held out. It now trains on partitions 1 to 3,
calibrates on 4, and partition 5 is never touched until the final test. Honestly the
number barely moved, AUC went 0.940 to 0.940, so the leak was not doing much, but I
would rather it be true than lucky.

The second one mattered more. src/fulldisk.py used to pick the best decision threshold
on the same partition it was scoring, which is just grading your own homework. Picking
the threshold on partition 4 and applying it to partition 5 gives TSS 0.71 instead of
0.75. That 0.04 was me marking my own exam.

One thing that bugs me: the threshold comes out at 0.24 on partition 4 but 0.12 on
partition 5. That is a big spread for one number, so I do not think the threshold is
very well pinned down, and I would not read too much into small TSS differences.

Then I lined it up against NOAA on live data, and it told me something humbling. On the
first graded day my forecast was way too low and a flare happened, and my live numbers
run well under NOAA's. I dug into why, and it is genuinely real. The
current regions just genuinely look quiet by the magnetic measurements my model reads,
while NOAA forecasts higher because their forecasters also use each region's recent
flare history and complexity, which a single magnetic snapshot does not have. So on
live data my model is currently more cautious than the experts, and the whole point of
the public scoreboard is to track that over time instead of hiding it.

One thing I want to be honest about. I first guessed the problem was bad calibration
and almost went and fixed that. Instead I measured, and the measurement showed my guess
was wrong, the historical forecast was already well calibrated. Checking before fixing
saved me from solving the wrong problem.

## Where this part lives (full disk and NOAA)

- src/fulldisk.py measures the whole-Sun daily forecast on held out history.
- src/live/noaa.py pulls NOAA's own forecast so the scoreboard can compare.
- results/fulldisk.json holds the historical full disk result.

## Step 9, does recent flare history help

NOAA partly beats me because their forecasters look at each region's recent flare
history, which my magnetic snapshot ignored. The data actually includes that history,
so I tested adding it. First I made sure those columns were not secretly the answer,
and they are not, a region that flares in the future does not have that future flare
written into its history columns. Then I trained the same model with and without it.

The result was honest and pretty small. Adding flare history left the TSS about the
same (around 0.83 either way), but it pushed the HSS from 0.20 up to 0.25, so fewer
false alarms. So history sharpens the model a little, but it does not break the
ceiling, which is the same lesson the four models taught me. And by itself it does not
close the live gap with NOAA, since that gap is more about the current regions
genuinely looking quiet than about one missing feature.

## Where this part lives (history experiment)

- src/history_experiment.py runs the with vs without comparison.
- results/history_experiment.txt has the numbers.

## Step 10, running it every day, and a bug the test caught

To grow a real track record without me doing anything, the forecast needs to run on its
own every day. I set that up with GitHub Actions, a free scheduler that runs in the
cloud once a day, issues the forecast, updates the scoreboard, and saves the log back to
the repo. No computer of mine has to be on.

Testing that daily job immediately caught a bug, which is the best argument for testing
it. One run forecast 100 percent, which is never a real answer. I traced it to a single
active region that had just rotated into view with only 3 measurements instead of the
usual 60. With that little data its summary numbers were garbage, and the model, which
only ever saw full 12 hour windows in training, got fooled into a near certain score.
The fix was simple, skip any region that does not have enough data yet, and never print
a literal 0 or 100 percent. After the fix the same day read 0.5 percent, which is
honest. The established regions really do look quiet to the magnetic measurements, even
though NOAA forecasts higher using region history my model ignores.

Later on the daily job died with a KeyError on HARPNUM, which looked scary but was not
my code's fault. JSOC had a 13 hour hole in the near real time SHARP series, so my query
came back with nothing. When that happens the drms library hands you an empty table with
no columns at all, and asking it to group by HARPNUM finds no such column. Retrying does
not help here, the server answers instantly and correctly with "I have nothing".

So now an empty result just returns no windows instead of blowing up. I also made it
refuse to log a forecast when zero regions actually got scored, because the full disk
math on an empty list gives 0.5 percent, and writing that to the log would look like a
real quiet day forecast when really I just had no data. A missing day in the log is more
honest than a made up one.

The job still grades the scoreboard on those days, since that part only reads the log and
does not need new Sun data. Then it exits with an error on purpose. I went back and forth
on that. A red X for a Stanford outage is not something I can fix, so it is sort of noise.
But the alternative is the job quietly doing nothing for a week and me never noticing my
forecast log stopped growing, which is worse. So it fails loudly and I get the email.

## Where this part lives (automation)

- src/daily.py is the once a day job, forecast plus scoreboard.
- .github/workflows/daily.yml is the cloud scheduler.
- .github/workflows/tests.yml runs the test suite on every push and pull request.

## Step 11, the model that was only allowed to say zero

The live forecast sat at exactly 0.5 percent for three straight weeks in August while
the Sun threw off M flares almost every other day and NOAA climbed from 10 percent to
55. Mine never moved. When I traced where the number actually came from, it turned out
the model was not being cautious about the live Sun. It could not say anything except
zero for the data it was seeing.

The calibration method I picked at the last retrain, isotonic regression, learns a
lookup table from raw model scores to probabilities, and the table it learned maps
every score below a certain cutoff to exactly 0. Every live region was scoring below
that cutoff. Part of the reason is that the near real time feed is missing some of the
measurements the model leans on. R_VALUE, one of the strongest known flare predictors,
comes back empty in the live series, and about 29 percent of the summary numbers going
in are blanks that get silently filled with the training average. So sixteen regions
went in, sixteen exact zeros came out, and combining sixteen zeros gives zero. The 0.5
percent in the log was never the model talking, it was just the floor I clamp to so
the log never shows a literal 0.

The fix was switching the calibration to Platt scaling, which squashes scores through
a smooth curve that can get very small but never reaches zero. I retrained, and the
honest cost showed up immediately. On the held out year the numbers dipped, AUC 0.94
to 0.92 and TSS 0.71 to 0.54 at the deployed threshold. So on paper the old model was
better. But the old model was also incapable of producing a nonzero forecast from the
data it actually gets every day, which makes its nicer paper numbers kind of
meaningless. I will take the slightly worse one that works. Its first live forecast
said 6 percent on a day NOAA said 55, so it is still the cautious one in the room, but
the number is finally real and it moves.

While I was in there I fixed a second thing I had been ignoring. Every measurement
JSOC sends comes with a QUALITY flag, and I had been downloading it daily and never
reading it, so about one row in six going into the model was one the instrument itself
had marked as bad. My first idea for the fix would have been a disaster, and I am glad
I measured before writing it. I was going to keep only rows whose flag is a clean
zero, and it turns out one hundred percent of live rows carry some routine
housekeeping flag, so that filter would have deleted every row of every day and the
forecast would have gone silent forever. The actual bad data marker is one specific
bit, so the filter checks that bit and nothing else. Same seam in the log as the last
retrain, forecasts before 2026-08-26 came from the old model, and every record says
which model made it.


One more note on the log. On 2026-08-27 GitHub never ran the scheduled job at all. Not
a failure, no red X, it just silently skipped the run, which they warn can happen when
their schedulers are busy. I waited about three and a half hours past the cron time and
then triggered the run by hand. It came out at 7.4 percent, which is the first forecast
the fixed model ever made on the real Sun outside my laptop.

I decided the rule for this before I ran anything, so I would not be picking based on
which number I liked better. If a single UTC day ever ends up with two forecasts in the
log, the first one stays and the second one gets deleted. Nothing about it depends on
how the forecast turned out.

And then it happened, on that same day. The run I triggered by hand landed at 04:29 UTC
and that is the 7.4 percent one. Right after it I retrained on the 17 keywords, and then
at 10:43 UTC a second run went off anyway, either GitHub's schedule finally waking up
nine hours late or a dispatch I fired twice, I honestly cannot tell which from the logs.
So 2026-08-27 had two forecasts and the rule said delete the later one. I did that on
2026-08-28 without looking at how either of them got graded.

It did sting a bit. The 04:29 one I kept came from the older model, and the 10:43 one I
deleted was the only Aug 27 forecast the retrained model ever made, so throwing it out
costs me a day off the new model's live record and pushes back when I can say anything
about it. Which is exactly the sort of argument the rule exists to shut down, so I
applied it anyway and the new model's track record just starts a day later. Every entry
in the log says which model made it, so you can check that yourself.

## Where this part lives (the zero bug)

- src/production.py now calibrates with Platt scaling instead of isotonic.
- src/live/fetch.py drops rows whose QUALITY flag has the bad data bit set.
- results/fulldisk.json has the held out numbers for the current model.

## Step 12, the seven features that were never there

After fixing the zero bug I still had one thing I could not explain. The model ranked
live days a lot worse than it ranked historical ones, and rescaling cannot cause that,
so something about the live data had to be different from the training data.

It was, and it is dumber than I expected. Seven of my 24 magnetic measurements were
arriving completely blank every single day. Not sometimes, not degraded, just gone.
TOTBSQ, TOTFZ, EPSZ, TOTFY, TOTFX, EPSY and EPSX. My first theory was that the near
real time feed is lower quality than the archival one, so I checked the archival series
too, and they are missing from that one as well. JSOC does not publish them at all.
They are Lorentz force quantities that the SWAN-SF authors computed themselves when
they built the dataset, so they exist in my training files and nowhere I can actually
reach.

So there was never a version of this that worked. Since the day I put it live, seven of
the 24 numbers going into the model were blanks, and my Standardizer quietly replaces
blanks with the training average. The model was reading "perfectly average sunspot" for
29 percent of its input on every region, every day. That is the ranking damage, and no
amount of recalibrating was ever going to touch it.

The fix is to train on the 17 measurements I can actually get. I expected to pay for
that, since dropping features usually costs you something. It did not. On held out data
the 17 feature model came out slightly ahead of the 24 feature one, AUC 0.9792 against
0.9767 at the instance level, and the full disk numbers went up too, AUC 0.922 to 0.928
and TSS 0.541 to 0.568. Those seven columns were not carrying information, they were
carrying a constant, and the model is better off without the distraction.

The part I was most nervous about was the wiring. Training reads a 144 number summary
built from all 24 parameters and picks out the 102 that survive, while the live path
builds 102 straight from a 17 column download. If those two orderings disagreed by even
one slot, every feature would land in the wrong place and nothing would crash, I would
just get quiet garbage forever. So there is now a test that builds both vectors from the
same input and checks they match. They agree to 2e-16.

I kept PARAMETERS at all 24 because that is a true statement about what is in SWAN-SF,
and the neural network experiments earlier in this project used all of them. The live
subset is a separate list. Nothing I already reported changed.

## Where this part lives (the missing features)

- src/load.py has NOT_SERVED_BY_JSOC and the LIVE_PARAMETERS subset.
- src/preprocess.py has live_columns, which picks the trainable subset out of a full
  SWAN-SF feature vector.
- tests/test_preprocess.py checks the training path and the live path agree exactly.

## Step 13, the day JSOC was down

On 2026-09-04 the scheduled run went off on time and still produced nothing. The job
sat there for eleven minutes and then died with a socket timeout. It never got as far
as the model. JSOC, the Stanford server that hands out the SHARP measurements, just
stopped answering, and my fetch code retries four times with waits of 30, 120 and 300
seconds, which adds up to almost exactly the eleven minutes the run took. So it tried
everything it knew how to try and the data was not there.

Nothing was wrong with my code. That is worth saying because my first instinct was to
go looking for a bug I had introduced.

But it did cost me a day of the live record, which I care about right now because I am
waiting on graded days to say anything about the retrained model. So I changed the
schedule. There is a second cron at 03:00 UTC now, two hours after the first one, and
both runs start by checking whether today already has a forecast in the log. If it
does, the run stops right there and does nothing. On a normal day the 03:00 run is a
ten second no-op.

That check turned out to fix two problems at once. The obvious one is the retry. The
other is the thing from 2026-08-27, where GitHub silently skipped the scheduled run
with no red X at all. A skipped run and a failed run look completely different in the
Actions tab but they leave the same hole in the log, and the second cron fills both.

It also means the one forecast per UTC day rule is now enforced by the code instead of
by me remembering it. Two runs can no longer both write a forecast for the same day,
which is the exact thing that happened to me on 2026-08-27 and cost me a day off the
new model's record.

If the 03:00 retry fails too, the workflow opens an issue on the repo saying which day
got lost and linking the failed run. GitHub emails me when an issue opens, so I hear
about it, and I end up with a written list of every day the data source let me down. I
left the first failure loud, it still goes red like it always did. I wanted to be able
to see a bad morning at a glance even on days where the retry saved it.

## Where this part lives (the retry)

- .github/workflows/daily.yml has both crons, the guard step, and the issue it opens
  when a day is genuinely lost.
- src/live/fetch.py is where the retries and the 45 second timeout live, unchanged.

## Step 14, the live score was bad and it was my own fault

Around the middle of September I finally had enough graded days to check whether the
retrained model was doing better live. It was doing worse. A lot worse. The old model
scored AUC 0.68 on its 60 live days, the new one scored 0.167 on 23 days. Anything
under 0.5 means it was ranking flare days below quiet days, which is worse than
guessing.

I chased three theories and all three were wrong, so I want to write them down because
being wrong three times in a row was most of the work here.

First I thought the calibration step had squashed the model's output range, because
live it never produced a number above 0.14 and the alarm threshold is 0.26. I loaded
the saved model and ran it on the held out year, and it still happily produces numbers
up to 0.97. Nothing was squashed.

Then I thought the live data must be miscalibrated. My training data was built from the
definitive SHARP product and my live code pulls the near real time one, and those are
genuinely two different data products. So I compared every live feature against the
training distribution. They all sit between the 19th and 73rd percentile. The live data
is sitting right in the middle of normal. That theory was dead too.

Then I thought maybe the Sun is just quiet right now and the model has never seen a
regime like it. I checked that by restricting the held out year to only the small quiet
regions, and the model still scored 0.93 there. Also dead.

What it actually was is embarrassing and much simpler. My training data has a median of
16 overlapping 12 hour windows per region per day, and when I score the held out year I
take the highest probability across all 16 of them. My live forecast looked at exactly
one window, whatever the last 12 hours happened to be, and used that. So the held out
number was best of 16 and the live number was best of 1. I had been comparing two
different things this whole time and calling the gap a model problem.

The fix is to make live do what the scoring does. It now pulls 24 hours instead of 12,
slides nine 12 hour windows back across that day, and takes each region's highest
probability. Every one of those windows still only uses data from before the forecast
goes out, so there is no cheating.

I replayed all 23 days through both versions before I changed anything. AUC went from
0.167 up to 0.45. So the fix is real and it is worth having, but I want to be clear
that 0.45 is still just coin flipping. This did not make the predictor work. It made
the live number mean the same thing as the offline number, which it did not before.

The honest caveat is that 23 days with only 3 flare days in them is not enough to
conclude much either way. I ran a permutation test on the original 0.167 and it comes
out around p 0.04, which is suggestive and nowhere near proof. The thing I trust here
is the windows bug, because that one does not depend on sample size at all.

Two other things I changed while I was in there. The forecast log now writes out all 17
feature values for every region, not just the top three probabilities. The only reason
this whole investigation was possible is that JSOC happened to still have September in
its near real time archive, and if it had been down I would have been stuck. Now the
numbers are in my own log. It costs about 2.4 MB a year. And the scoreboard reports AUC
now, which is a little silly given AUC is the number this entire step was about and I
was computing it by hand in a scratch file every time.

## Where this part lives (the windows fix)

- src/live/fetch.py keeps the timestamp of every record now, and pulls 24 hours.
- src/live/forecast.py has best_window, which is the slide and take the max part.
- src/metrics.py has auc, and src/live/scoreboard.py reports it.

## Step 15, adding up the regions wrong

After Step 14 I went looking for a second bug of the same shape, because if I got the
windows wrong I could easily have gotten something else wrong too. The thing I suspected
was the region count. My training days have a median of about 4 or 5 active regions in
them, and live I am looking at 10 to 25. The way I combine regions into one whole-Sun
number is noisy-OR, which is one minus the chance that every region stays quiet, and that
formula gets bigger just from having more regions in it. So I thought the live number was
being inflated by region count alone.

It is not. I tested it by padding every day in the held out year with fake extra quiet
regions until each day had 10, then 15, then 20, then 25, and the score barely moved,
0.928 to 0.931. That theory was wrong, which is now the fourth wrong theory in a row.

But the test I wrote to check it also compared noisy-OR against other ways of combining
regions, and that turned up something real. Just taking the single strongest region on
the disk beats noisy-OR at everything I care about:

```
                     AUC p4   AUC p5   threshold   TSS p5   mean forecast
noisy-OR (old)        0.829    0.928        0.26    0.568           0.107
noisy-OR top 3        0.831    0.941        0.18    0.658           0.095
strongest region      0.827    0.948        0.10    0.726           0.068
```

Partition 5 is the held out year, 823 days with 66 flare days in it, and the actual flare
rate there is 0.080. So the old way was forecasting 10.7 percent on average when the truth
was 8.0 percent, and the new way says 6.8 percent, which is closer. The three options are
basically tied on partition 4, which is where I pick the threshold, so this is not me
fishing on the partition I tune against.

I bootstrapped the AUC difference on the held out year to check I was not reading noise.
The gain is 0.0196 with a 95 percent interval of 0.0073 to 0.0332, and the strongest
region version wins in 99.9 percent of resamples. That is about as clear as anything I
have measured on this project.

Why would throwing away information make it better? Because noisy-OR assumes the regions
are independent and they really are not. They sit on the same Sun during the same level
of activity. So it double counts, it piles up a little probability from every quiet region
on the disk, and it drifts above the truth. Taking the worst region throws that pile away.
It also makes the number completely independent of how many regions JSOC decides to serve
me that day, which is the single biggest difference between my training days and my live
days.

A nice side effect: my threshold got much more stable. It used to be 0.26 on partition 4
and 0.12 if I refit it on partition 5, which is a big gap and I had written that up as a
weakness I was just going to keep disclosing. Now it is 0.10 against 0.08. The TSS I
actually get, 0.726, is close to the 0.747 I would get by cheating. I did not set out to
fix that and it fell out anyway.

I did not retrain anything for this. The model weights are byte for byte the same, I only
recomputed the threshold under the new way of combining regions, which is what
retune_threshold in src/production.py does.

The annoying part is what this does to the live scoreboard. Between Step 14 and Step 15 I
changed both how the windows are chosen and how the regions are combined, so a forecast I
issued in August and a forecast I issue tomorrow are not the same measurement and it would
be flattering myself to average them together. Every forecast now gets stamped with which
scoring produced it, and the public scoreboard only summarises the current one. The 84
older graded days stay in the log, they just do not get counted. So the track record on
the site goes back to zero today, which looks bad and is correct.

I changed it now on purpose rather than waiting. The live record had just restarted anyway
because of Step 14, so restarting it twice in two days costs me nothing, while shipping it
a month from now would have thrown away a month.

## Where this part lives (the aggregation)

- src/fulldisk.py, daily_scores now returns the strongest region per day.
- src/live/forecast.py does the same thing live, and stamps SCORING onto every forecast.
- src/production.py has retune_threshold, which repicks the threshold without retraining.
- src/live/scoreboard.py only summarises forecasts under the current scoring.

## Where it stands now

That is the whole build. The short version of where it landed:

- Four models, from a one line logistic regression to a hand built LSTM, all around
  TSS 0.83 on an honest time based split. The from scratch nets match PyTorch to
  machine precision.
- The whole-Sun daily forecast scores AUC 0.95 and TSS 0.73 on a genuinely held out
  year, with the threshold picked before ever looking at that year, and it is
  calibrated. It used to read better than that on paper, and the version that read
  better could not function on live data at all, which is Steps 11 and 12.
- A live system pulls the real Sun every day, forecasts, and grades itself against both
  reality and NOAA. Right now it runs more cautious than NOAA, and the public scoreboard
  tracks that honestly over time.
- Live, it still has no demonstrated skill. After fixing the windows bug in Step 14 it
  sat around AUC 0.45 over 23 graded days, which is a coin flip, and after Step 15 the
  live count is back to zero and building again. The offline numbers
  are real and the live numbers are real and they do not agree yet, and I would rather
  say that plainly than quietly report the 0.95.

Honestly the part I care about most is that habit. Every time a result
looked too good, I dug in and it turned out to be a leak or an artifact or the metric
flattering me, and I fixed it instead of reporting it. The deeper technical version of
all this is in docs/paper.md.

