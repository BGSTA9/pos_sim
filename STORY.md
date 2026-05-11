# How POS Finds a Heartbeat in a Webcam — The Plain-English Story

> *A friendly explainer for anyone who isn't a signal-processing engineer.*
> Read time: ~6 minutes. No equations required.

---

## The hook

You're sitting in front of your laptop. The camera is on. Could that
webcam, with no contact at all, tell you your heart rate?

It can. And not because of AI or anything fancy. **Because of physics.**

This folder is a simulation that proves it — in clean math, on
synthesized data. This page tells the story of what's actually
happening inside.

---

## The tiny color change

Every time your heart beats, a wave of blood pushes through the
capillaries just under your skin. Hemoglobin — the red stuff in
blood — happens to absorb green light a little more strongly than red
or blue. So your skin gets a tiny bit **less green** the moment a pulse
passes through it.

How tiny? About **one part in a thousand**. If your forehead reads
G = 130 to the camera, it might dip to G = 129.87 and bounce back, sixty
times a minute. Your eyes can't see it. The camera can — barely. But
the signal is real.

If we could pull *just that ripple* out of the video, we'd know exactly
when each heartbeat happened.

---

## The catch: everything else is louder

In real life, that 0.1 % green ripple is buried under a pile of much
bigger nonsense:

- **You move your head.** The angle of the light changes. The whole
  image gets brighter and darker by a few percent.
- **Shadows drift.** Someone walks past a window. A cloud moves. The
  fluorescent bulb above you flickers.
- **You sweat, or your skin glints.** A bright reflection — a
  **specular highlight** — moves across your forehead. That's not blood;
  it's just the lamp's own color bouncing off you.

These distractions are *thousands of times louder* than the pulse. It's
like trying to hear someone whisper while a freight train rolls by.

So the question is: how do we mute the freight train without also
muting the whisper?

---

## The trick: every kind of noise has its own *color*

Here's the insight POS exploits.

The different sources of color change *aren't all the same color*.

- **Shadows and motion** dim or brighten all three channels together.
  If the lighting drops 5 %, R, G *and* B all drop 5 %.
- **Specular glints** are the color of the *lamp*. Under normal
  white-ish indoor light, they're roughly grayish-white.
- **The pulse** has a very specific color signature: it shows up most
  in Green, less in Blue, least in Red. Always in that order.

Each of those is a "direction" in color space — an arrow pointing
somewhere in the 3-D box made by the R, G, B axes. And critically,
those arrows point in *different* directions.

That means we can **cancel** the noise arrows by clever geometry, not
by filtering. It's not "smooth the signal" — it's "look at the signal
from the right angle so the noise disappears".

---

## What the simulation actually does

We don't have a real video. We build a *fake* one — but not with random
numbers. We build it using the **actual physics of skin reflection**,
exactly as Wang et al. wrote it down in their 2017 paper. Three pieces.

### Piece 1 — `SyntheticDataGenerator`: the fake face

This creates a 30-second-long synthetic webcam recording of someone
with a 72 BPM heart rate. It does this by literally adding together:

1. **A static skin tone** — the baseline R, G, B of an average face
   under indoor light.
2. **A pure sine wave at 1.2 Hz** (= 72 beats per minute), painted into
   the green channel hardest, blue second, red least. That's the pulse.
3. **A slow wandering brightness drift** — the "moving shadows".
4. **A faster, motion-band flicker** in the lamp's color — the
   "specular glints".
5. **A pinch of digital sensor noise** for realism.

And here's the cruel part: we crank items (3) and (4) up to be
**ten times louder than the pulse**. This is the worst-case stress
test. If the algorithm can still find a 72 BPM heartbeat in *that*
disaster, we know it'll handle real video.

The output is a wiggly 3-channel signal that, plotted, looks like
meaningless squiggles ([fig1](figures/fig1_raw_rgb.png), top panel).
The pulse is in there, but completely invisible to the eye. The bottom
panel of that figure shows what's *really* inside — including the tiny
green pulse line dwarfed by the purple intensity drift and the orange
specular drift.

### Piece 2 — `POSProcessor`: the de-mixer

This is the heart of the whole project. It takes the mess from Piece 1
and pulls the pulse out. It works in short overlapping chunks (each
about 1.6 seconds, ~2 heartbeats long), and for each chunk it does four
geometric moves:

**Move 1 — Erase the baseline.** Divide every sample by the average of
the window. After this step, you're not looking at "skin color"
anymore — you're looking at *changes from baseline*. The boring static
skin tone becomes irrelevant. Every face — light, dark, anywhere on the
spectrum — collapses to the same neutral starting point.

**Move 2 — Cancel the shadows.** Project the trace onto a 2-D plane
that's perpendicular to the "all three channels move together"
direction. Because shadow-style brightness changes affect R, G, B
*equally*, they live entirely along that direction. So the projection
erases them. We've gone from 3 channels to 2 cleaner ones, named
`S₁` and `S₂`. Their formulas are stupidly simple:

> `S₁` = Green − Blue
> `S₂` = Green + Blue − 2 × Red

No machine learning. No neural network. Just additions and
subtractions of pixel values. That's the entire algorithm at the pixel
level.

**Move 3 — Cancel the glints.** The projection was *deliberately
chosen* so that:

- The **pulse** shows up moving the **same way** in both `S₁` and `S₂`.
- The **specular noise** shows up moving in **opposite** ways.

So if we just add `S₁` + `S₂`, the pulse parts pile up (louder!) and
the specular parts cancel out toward zero (gone!). To make the
cancellation exact, we measure how loud `S₁` is, measure how loud `S₂`
is, and scale `S₂` so they match. That ratio is called *alpha*
(`α = σ(S₁) / σ(S₂)`). The kind of trick that sounds obvious after you
hear it.

**Move 4 — Stack the chunks.** Slide the window forward by one frame,
redo all the moves, and add the new little pulse fragment on top of the
old one. After a full recording, every moment has been processed by
dozens of overlapping windows. The few unlucky windows where the math
got fooled by a noise burst get averaged out by all the lucky ones
around them.

The output is a single 1-D waveform `H(t)` — the recovered heartbeat.

### Piece 3 — `SignalAnalyzer`: the heart-rate reader

`H(t)` is *almost* the pulse but still has some out-of-band crud. So:

1. **Filter** out anything that couldn't possibly be a human heart
   rate. The band that survives is 0.7 to 4.0 Hz — that's 42 to 240
   beats per minute. (A resting yogi sits near the bottom; a sprinting
   athlete near the top.)
2. **Fourier transform** the result — i.e., ask: "what rhythms are
   present?" The biggest peak in the answer *is* the heart rate.
3. **Polish that peak.** The Fourier method's frequency resolution is
   finite — one of its measurement bins is about 7.5 BPM wide. Too
   coarse for medicine. We fit a small parabola across the top of the
   peak to nail down its exact location, getting accuracy down to
   roughly 1 BPM.

---

## The proof

When we run the whole pipeline on the synthetic mess (10× noise!), the
algorithm reports:

> **Ground truth: 72.00 BPM**
> **Detected:     72.72 BPM**
> **Error:        0.72 BPM (about 1 %)**

Across ten different random simulations, the worst error we *ever* see
is 0.80 BPM. The recovered pulse waveform locked in phase with the true
one ([fig2](figures/fig2_recovered_pulse.png)). The frequency spectrum
has a single clean spike right where it should be
([fig3](figures/fig3_spectrum.png)) — no leakage from the 10×-louder
noise.

In a deliberately worst-case fake video, three short stages of
plain linear algebra recover the heart rate to within 1 % accuracy.
That's the result.

---

## Why this matters

Every smartphone, every laptop, every webcam can — in principle — do
this. No new hardware, no contact, no electrodes. The applications
people are already chasing:

- A **baby monitor** that watches an infant's heart rate without
  attaching any wires to a tiny body.
- A **telehealth app** that takes vitals during a video call.
- **Driver-fatigue detection** in a car.
- **Patient monitoring** in burn wards where electrodes can't touch the
  skin.

This folder isn't the product. It's the **proof** — done in clean math
on synthesized data — that the algorithm at the core of all those
applications actually works the way the paper says. Once that core is
trustworthy, the rest is engineering: face detection, region tracking,
robustness across skin tones, motion artifacts, all of it.

But it starts with the math doing what it's supposed to. And that's
what this simulation shows.

---

## See also

- [architecture.png](figures/architecture.png) — the pipeline diagram,
  one picture, end to end.
- [README.md](README.md) — the technical reference: equations,
  derivations, code layout.
- [notebooks/demonstration.ipynb](notebooks/demonstration.ipynb) — run
  through the whole thing cell by cell.
