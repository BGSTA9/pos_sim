> **System Prompt & Role:**
> You are an elite Computational Mathematician and Signal Processing Engineer. Your task is to build a production-grade, Object-Oriented Python simulation of the POS (Plane-Orthogonal-to-Skin) rPPG (remote Photoplethysmography) algorithm, as defined by Wang et al.
> Your code must be highly optimized, heavily commented, and adhere strictly to the mathematical physics of the Dichromatic Reflection Model. Do not use black-box libraries for the core algorithm; implement the linear algebra using NumPy.

> **Project Architecture & Requirements:**
> I need you to build a complete simulation pipeline consisting of three main Object-Oriented classes.

> **Class 1: `SyntheticDataGenerator` (The Physics Simulator)**
> Before we process real video, we need a synthetic RGB signal generator to validate the math.
> * **The Math:** Implement the Linear Mixture Model: $C(t) = I_0 \cdot (1+i(t)) \cdot (u_c \cdot c_0 + u_s \cdot s(t) + u_p \cdot p(t)) + v_n(t)$
> * **Components:**
> * Generate a ground-truth heartbeat $p(t)$ (a sine wave at 72 BPM / 1.2 Hz).
> * Add a massive low-frequency intensity fluctuation $i(t)$ (simulating shadows/flickering light).
> * Add a specular glare fluctuation $s(t)$ (simulating head movement).
> * Assume the blood volume pulse vector ranking is $G > B > R$.
> * Add white Gaussian noise $v_n(t)$ for camera sensor rounding errors. 
> * **Output:** Return a simulated $N \times 3$ array representing the spatial-averaged RGB trace of a face over time.
> 
> 
> **Class 2: `POSProcessor` (The Core Engine)**
> This is the implementation of Algorithm 1 from the POS paper. It must process the RGB trace using a sliding window.
> * **Windowing:** Implement a sliding window of length $l=32$ frames (assuming 20 fps, or adapt to the sampling rate to ensure at least one full cardiac cycle).
> * **Step 1: Temporal Normalization:** For each window, divide the RGB channels by their temporal mean to map the static skin-tone to the $[1, 1, 1]$ vector.
> * **Step 2: Orthogonal Projection:** Multiply the normalized window by the exact POS projection matrix to eliminate intensity variations:
> $P = \begin{pmatrix} 0 & 1 & -1 \\ -2 & 1 & 1 \end{pmatrix}$
> This creates two signals: $S_1 = G_n - B_n$ and $S_2 = G_n + B_n - 2R_n$.
> * **Step 3: Alpha-Tuning:** Calculate the standard deviation of $S_1$ and $S_2$ to find the tuning parameter $\alpha = \sigma(S_1) / \sigma(S_2)$.
> * **Step 4: Signal Combination:** Calculate the pulse $h = S_1 + \alpha \cdot S_2$.
> * **Step 5: Overlap-Add:** Use an overlap-add methodology to reconstruct the continuous pulse signal $H(t)$ from the sliding windows, ensuring you subtract the mean of the window before adding it to the global signal.


> **Class 3: `SignalAnalyzer` (The Grader)**
> This class will clean the output of the `POSProcessor` and extract the final BPM.
> * **Filtering:** Implement a zero-phase Butterworth bandpass filter. The cutoff frequencies should be 0.7 Hz (42 BPM) and 4.0 Hz (240 BPM).
> * **Frequency Analysis:** Implement a Fast Fourier Transform (FFT) or Welch's method (Power Spectral Density) to find the dominant frequency peak.
> * **Validation:** Compare the extracted BPM to the ground-truth 72 BPM from the `SyntheticDataGenerator`.
> 
> 
> **Deliverable:**
> Write the complete Python code for these three classes. Then, provide a `main()` execution block that initializes the synthetic data, runs it through the POS processor, filters it, and uses Matplotlib to plot three things:
> 1. The messy synthetic RGB signal.
> 2. The extracted clean pulse wave $H(t)$ overlaid with the ground truth $p(t)$.
> 3. The frequency spectrum showing the dominant BPM peak.


Note: In the "SyntheticDataGenerator" make the "Noise" (the motion and intensity) about 10 times stronger than the pulse. This creates a "Worst-Case Scenario" simulation. If my code can still find the 72 BPM peak in that mess, this project will be a masterclass in signal recovery.

Following is the "Folder Structure" for your ease:
"
pos_sim/
├── src/
│   ├── __init__.py
│   ├── generator.py      <-- Class 1: Physics Engine
│   ├── processor.py      <-- Class 2: The POS Logic
│   └── analyzer.py       <-- Class 3: FFT/Metrics
├── notebooks/
│   └── demonstration.ipynb <-- The "Narrative" (Charts + Math)
├── main.py               <-- Standalone entry point
├── requirements.txt      <-- Dependency list
└── README.md             <-- The Mathematical Appendix we wrote
"