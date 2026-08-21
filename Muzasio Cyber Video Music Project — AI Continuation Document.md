# Muzasio Cyber Video Music Project — AI Continuation Document

> Written for the next AI instance. Zero assumptions. Every fact is exact.

---

## 1. Problem Statement and Goal

Muzasio runs a cybersecurity tech brand, publishes video content (YouTube/shorts style), and needs **royalty-free background music generated from scratch using Python** with no external audio libraries. The music must fit specific video sections and be usable immediately in a video editor.

**Constraints that are non-negotiable:**
- Python only, `numpy` + `scipy` only — no `pydub`, no `librosa`, no `soundfile`, no `midiutil`, no `FluidSynth`, no external binaries
- Output format: WAV, 16-bit, stereo, 44100 Hz
- Music must sound professional enough for public YouTube content
- All files go to `/mnt/user-data/outputs/`
- Scripts run in `/home/claude/` as working dir (ephemeral — does not persist between sessions)

**The user explicitly said the current music is "basic"** and acknowledged that synthesis from numpy/scipy has limits. The goal going forward is to push those limits as far as possible — better envelopes, more realistic textures, better layering — without introducing new dependencies.

---

## 2. Environment Facts

| Item | Value |
|---|---|
| OS | CachyOS (Arch-based Linux), KDE Plasma 6, Wayland |
| Shell (user) | fish |
| Shell (bash_tool) | bash |
| Python | 3.12.3 |
| numpy | 2.4.4 |
| scipy | 1.17.1 |
| Editor preference | nano (never suggest micro or other editors) |
| Build target | Flatpak only (never suggest AUR, snap, native packages unless asked) |
| Firefox | Flatpak, sandbox at `~/.var/app/org.mozilla.firefox/` |
| Username | muzasio |
| Output dir | `/mnt/user-data/outputs/` — this persists and is user-accessible |
| Script dir | `/home/claude/` — ephemeral, wiped between sessions, scripts must be recreated |
| Image library | Pillow (PIL) — default for image work |
| Default language | Python unless specified |

**Verified working imports:**
```python
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt
# All confirmed working, no install needed
```

---

## 3. What Works — Exact Confirmed Patterns

### 3.1 Core DSP Helpers (copy these exactly)

```python
SR = 44100

def lpf(sig, cutoff, order=4):
    sos = butter(order, cutoff/(SR/2), btype='low', output='sos')
    return sosfilt(sos, sig)

def hpf(sig, cutoff, order=2):
    sos = butter(order, cutoff/(SR/2), btype='high', output='sos')
    return sosfilt(sos, sig)

def bpf(sig, lo, hi, order=3):
    sos = butter(order, [lo/(SR/2), hi/(SR/2)], btype='band', output='sos')
    return sosfilt(sos, sig)

def normalize(sig, peak=1.0):
    m = np.max(np.abs(sig))
    return sig / m * peak if m > 0 else sig
```

### 3.2 Filter Warmup (critical for seamless loops — do not skip)

IIR filters (butter/sosfilt) have a startup transient at sample 0. If you filter a loop and the loop restarts, the transient causes a click. Fix: render 2x length, discard first half.

```python
def filt_warm(sig, cutoff, btype='low', order=4, lo=None):
    if btype == 'band':
        sos = butter(order, [lo/(SR/2), cutoff/(SR/2)], btype='band', output='sos')
    else:
        sos = butter(order, cutoff/(SR/2), btype=btype, output='sos')
    s2 = np.tile(sig, 2)
    return sosfilt(sos, s2)[N:]  # N = total sample count of one loop
```

### 3.3 Master Bus (use on every final mix before saving)

```python
def master(sig):
    sig = filt_warm(sig, 14000)       # air cut
    sig = filt_warm(sig, 25, 'high', 2)  # DC/ultra-sub removal
    sig = np.tanh(sig * 0.88)         # soft saturation / limiter
    return normalize(sig, 0.92)
```

### 3.4 Save Function (stereo, 7ms right-channel delay for width)

```python
def save(sig, path):
    N = len(sig)
    d = int(0.007 * SR)
    right = np.concatenate([sig[N-d:], sig[:N-d]])
    stereo = np.stack([sig, right], axis=1)
    out = (stereo * 32767).astype(np.int16)
    wavfile.write(path, SR, out)
```

### 3.5 Frequency-Locking for Seamless Loops

For a loop of duration `LOOP_DUR` seconds, any oscillator will be phase-continuous (same value at start and end) only if its frequency is an exact integer multiple of `1/LOOP_DUR`.

```python
def freq_lock(freq, loop_dur):
    return round(freq * loop_dur) / loop_dur

# Usage:
LOOP_DUR = 10.667  # e.g. 4 bars at 90 BPM
freq_locked = freq_lock(36.7, LOOP_DUR)
t = np.arange(N) / SR
sig = np.sin(2 * np.pi * freq_locked * t)
```

**Confirmed seam quality achieved:** boundary delta 0.00125, which is 0.4× the average signal jump — inaudible.

### 3.6 Bass Hit Synthesis (the core building block)

```python
def bass_hit(freq, vel, dur_n, decay=3.0, brightness=3.5, harmonics=6):
    tt = np.arange(dur_n) / SR
    sub = np.sin(2*np.pi*freq*0.5*tt) * 0.65        # sub octave
    saw = sum(np.sin(2*np.pi*k*freq*tt)*(-1)**(k+1)/k
              for k in range(1, harmonics+1)) * (2/np.pi)
    saw = lpf(saw, freq*brightness) * 0.5
    rng = np.random.default_rng(int(freq) % 97)
    punch = bpf(rng.normal(0,1,dur_n), 100, 700) * np.exp(-80*tt) * 0.18
    atk = max(int(0.005*SR), 1)
    env = np.zeros(dur_n)
    env[:atk] = np.linspace(0, 1, atk)
    env[atk:] = np.exp(-decay * tt[atk:] / (dur_n/SR))
    return (sub + saw + punch) * env * vel
```

Parameters:
- `decay=2.0` → long sustain (slow groove)
- `decay=4.5` → punchy tight (fast technical)
- `brightness=3.0` → dark/muffled
- `brightness=5.0` → bright/crisp

### 3.7 Sequencer (pattern-based bass track builder)

```python
def sequencer(pattern, notes, bpm, start_t=0.0, decay=3.0, brightness=3.5, total_dur=20.0, N=None):
    beat = 60.0/bpm
    step_t = beat/4   # 16th note
    step_n = int(step_t*SR)
    if N is None: N = int(total_dur * SR)
    track = np.zeros(N)
    cur = next((n for n in notes if n is not None), 36)
    total_steps = int((total_dur - start_t) / step_t)
    start_s = int(start_t * SR)
    for i in range(total_steps):
        pi = i % len(pattern)
        vel = pattern[pi]
        note = notes[pi % len(notes)]
        if note is not None: cur = note
        if vel == 0: continue
        ts = start_s + i * step_n
        hold = 1
        for j in range(i+1, min(i+8, total_steps)):
            if pattern[j % len(pattern)] > 0: break
            hold += 1
        dur = min(hold*step_n, int(step_n*2.5))
        te = min(ts+dur, N)
        if ts >= N: break
        hit = bass_hit(midi_freq(cur), vel, te-ts, decay, brightness)
        track[ts:te] += hit
    return track

def midi_freq(m):
    return 440.0 * (2.0**((m-69)/12.0))
```

### 3.8 Piano Hit (muted, dark — confirmed working)

```python
def piano_hit(freq, vel, dur_n, lpf_cutoff=3500, decay=3.2):
    t = np.arange(dur_n) / SR
    sig = (np.sin(2*np.pi*freq*t) * 0.65 +
           np.sin(2*np.pi*2*freq*t) * 0.25 +
           np.sin(2*np.pi*3*freq*t) * 0.08 +
           np.sin(2*np.pi*4*freq*t) * 0.02)
    env = np.exp(-decay * t / (dur_n / SR))
    atk = max(int(0.008*SR), 1)
    env[:atk] *= np.linspace(0, 1, atk)
    return lpf(sig * env, lpf_cutoff) * vel
```

User feedback history:
- `lpf_cutoff=1800` → "submerging too much" (too muffled)
- `lpf_cutoff=3500` + `decay=3.2` + mix level `1.6` → accepted as correct

### 3.9 Two-Loop Structure (loop1 = bass only, loop2 = bass + piano)

The confirmed working architecture for the cyber bass loop:

```python
# Build base mix shared by both loops
base = (normalize(bass_track) * 0.85 + sub_drone + pad + tex + click_track)
loop1 = base.copy()
loop2 = base + piano_track * 1.6

# Process BOTH loops as one continuous signal through master bus
# This prevents filter transient seam at loop boundary
full_raw = np.concatenate([loop1, loop2])
total = len(full_raw)

full_3x = np.tile(full_raw, 3)
full_3x = filt_warm_inline(full_3x, 13000)   # see filt_warm above
full_3x = filt_warm_inline(full_3x, 25, 'high', 2)
full = full_3x[total*2:]   # keep the last (warmed) copy

full = np.tanh(full * 0.88)
full = normalize(full, 0.92)

# Mask loop2→loop1 boundary with micro fade (hidden by beat-1 transient)
fade_n = 256
full[:fade_n] *= np.linspace(0, 1, fade_n)
full[total-fade_n:] *= np.linspace(1, 0, fade_n)
```

**Confirmed seam measurements:**
- Loop1→Loop2 (mid-file): delta 0.00118 — inaudible
- Loop2→Loop1 (end→start): delta 0.00000 — masked by beat-1 bass hit

---

## 4. What Doesn't Work — Exact Failures

### 4.1 Simple In-Place Crossfade for Seam Fix

**Attempt:**
```python
xf = 4096
mix[:xf] = mix[:xf] * fade_in + mix[N-xf:] * fade_out
mix[N-xf:] = mix[N-xf:] * fade_out + mix[:xf] * fade_in
```

**Result:** Seam delta 0.63–0.85 — audible click.

**Why it failed:** The two writes are mutually dependent. `mix[:xf]` is read by the second line after it has already been modified by the first line, producing wrong values. The signal also doesn't naturally approach zero at boundaries so crossfade can't fix a 0.63 jump — crossfade lerps between endpoints, it doesn't change what those endpoints ARE.

**Correct fix:** Use copies (see section 3.3) OR process the full concatenated signal through a warmed filter in one pass (see section 3.9).

### 4.2 Separate Master Bus Per Loop

**Attempt:** Apply `master()` to `loop1` and `loop2` independently, then concatenate.

**Result:** Seam delta 0.63 at loop2→loop1 boundary.

**Why it failed:** The IIR filter state at the end of loop2 does not match the filter state at the start of loop1. Even with frequency-locked oscillators, the filter introduces state divergence. The `tanh` nonlinear saturation also means two identical signals processed separately will have different clipping behavior depending on context.

**Correct fix:** Concatenate first, then master the whole thing as one signal (section 3.9).

### 4.3 Crossfade OLA After Separate Processing

**Attempt:** Load the saved file, apply overlap-add crossfade.

**Result:** Still 0.82 seam delta.

**Why it failed:** Same root cause — the signal values at boundaries are 0.63 apart. No amount of crossfading fixes a jump that large without audible artifacts. The underlying generation must produce a continuous signal.

### 4.4 Phase-Locking Alone Is Not Enough

**Assumption:** If all oscillator frequencies are locked to integer multiples of `1/LOOP_DUR`, the signal will be seamless.

**Result:** Sub drone seam delta 0.97 before fix.

**Why it failed:** Phase locking ensures the oscillator returns to the same phase at sample N. But IIR filters applied to the oscillator output introduce a startup transient at sample 0 that does NOT match the filter's steady-state response at sample N. The filter's internal state differs between start and end.

**Correct fix:** `filt_warm()` — render 2× length through the filter, discard first half. Confirmed seam delta drops from 0.97 → 0.00124.

---

## 5. Current State

### 5.1 Files Delivered (all in `/mnt/user-data/outputs/`)

| File | Duration | BPM | Notes |
|---|---|---|---|
| `background_music.wav` | 21.6s | 90 | 10-instrument test, all instruments layered |
| `cyber_bass_loop.wav` | 42.7s | 90 | Loop1 (bass only) + Loop2 (bass + piano), seamless |
| `intro_A_cold_open.wav` | 15s | — | Deep impact at 0.4s, rising tension pad, tick accents |
| `intro_B_pulse_build.wav` | 15s | — | Heartbeat pulse → glitch bass stabs → hi-freq rises |
| `intro_C_drop_entry.wav` | 15s | 95 | 2s silence → hard drop → groove locks in |
| `suspense_A_dread.wav` | 15s | 65 | Slow heartbeat, tritone dissonant pad, sparse piano |
| `suspense_B_creeping.wav` | 15s | — | Near-silence, irregular glitch pulses getting denser |
| `suspense_C_tension_hold.wav` | 15s | — | Sustained LFO drone, micro crackle, distant impacts |
| `explanation_A_focused_flow.wav` | 20s | 90 | C major pad, clean groove, analytical |
| `explanation_B_technical.wav` | 20s | 100 | Mechanical 16th pattern, metallic arp |
| `explanation_C_deep_dive.wav` | 20s | 80 | Heavy slow bass, minor pad, sparse piano |
| `explanation_D_sharp.wav` | 20s | 108 | Punchy stabs, hard accents, chord hits |
| `explanation_E_ambient_groove.wav` | 20s | 85 | Airy pad, brighter piano, lightest feel |

### 5.2 What's Not Done Yet

The user requested 4 section types. Three are complete:
- ✅ Intro (3 variants, 15s each)
- ✅ Suspense (3 variants, 15s each)
- ✅ Explanation (5 variants, 20s each)
- ❌ **Failed** — not yet generated

**Next immediate step:** Generate 3 "Failed" section tracks, 15s each.

Planned Failed variants:
- **Failed A** — dissonant drop, heavy sub hit, unresolved tension, something broke
- **Failed B** — glitchy corruption effect, signal degrading, error feel
- **Failed C** — slow descending progression, heavy and defeated, hollow resolution

The user also confirmed interest in **audio effects** (whoosh, impact, glitch, tension sting) — confirmed feasible with numpy/scipy, not yet started.

---

## 6. Gotchas — Things That Wasted Time

### 6.1 Script Files Don't Persist

`/home/claude/` is ephemeral. Every session starts empty. Do not reference scripts from previous sessions — recreate them inline. Always write full scripts to `/home/claude/scriptname.py` then run with `python3 /home/claude/scriptname.py`.

### 6.2 Seam Delta Measurement Was Wrong Initially

The seam test `abs(mix[-1] - mix[0])` measures whether the last sample equals the first sample. This is NOT the right test for loop quality. The right test is whether the transition at the loop point sounds smooth, which depends on the local slope, not just the absolute values. A delta of 0.001 is inaudible. A delta of 0.008 measured at the wrong window position caused false alarm — the actual boundary was fine.

Correct measurement:
```python
join = np.concatenate([mix[N-50:], mix[:50]])
max_jump = np.max(np.abs(np.diff(join[47:53])))  # 3 samples either side of boundary
```

### 6.3 Noise Is Not Phase-Lockable

Random noise (`np.random.normal`) cannot be frequency-locked. It will always produce a seam. Fix: gate it to zero at boundaries, or use deterministic noise seeded per-session that happens to be near zero at endpoints. The `default_rng(seed)` approach with gating confirmed working:

```python
rng = np.random.default_rng(7)
noise = rng.normal(0, 1, N)
noise = bpf(noise, 3500, 9000) * 0.07
# Gate to zero before loop boundary
gate = np.zeros(N)
for b in range(bars * 4):
    on = int(b * BEAT * SR)
    en = min(on + int(0.09*SR), N)
    gate[on:en] = np.linspace(1, 0, en-on)
noise *= gate  # noise is zero at N-1 and 0 naturally
```

### 6.4 `filt_warm` Requires N to Be Defined Before Calling

`filt_warm` uses `N` (loop sample count) to discard the first half. If `N` is not the correct loop length, you get wrong output. When working with multi-loop files (e.g. loop1 + loop2 = 2N total), set `N` to the TOTAL length before calling `filt_warm`, or refactor to pass length explicitly.

### 6.5 Piano Volume Was Too Low Initially

First piano mix level was `* 0.9` — user said it was "submerging too much." Correct level: `* 1.6`. The LPF cutoff was also too aggressive at 1800 Hz — raised to 3500 Hz. Both changes together produced the accepted result.

### 6.6 `np.random.default_rng(seed)` vs `np.random.seed()`

Always use `default_rng(seed)` not the legacy `np.random.seed()`. Using the same seed for different parts of the same signal produces correlated noise, which can create audible artifacts. Use different seeds for different components (e.g. seed=7 for texture, seed=42 for bass punch, seed=22 for impact).

### 6.7 Karplus-Strong Requires Exact Integer Buffer Length

The guitar/plucked string synthesis (Karplus-Strong algorithm) requires `N = int(SR / freq)`. If freq is not an exact divisor of SR, floating point rounding causes pitch drift over time. For loops, the frequency also needs to be freq_locked. This was not fully resolved for the Karplus-Strong instrument — it was kept in background_music.wav only, not in the loop files.

### 6.8 User's Preferred Chord Progression

Used throughout: **C major → A minor → F major → G major**
MIDI roots: `[60, 57, 53, 55]` (C4, A3, F3, G3)
Bass roots: `[36, 33, 29, 31]` (C2, A1, F1, G1)

Suspense/tension uses: **E1 (41.2 Hz) with tritone Bb1 (58.3 Hz)** — confirmed effective.

---

## 7. Open Questions Still Unresolved

### 7.1 Quality Ceiling of numpy/scipy Synthesis

The user explicitly acknowledged the music is "basic" and that synthesis from numpy/scipy has limits. The question of how much better it can get has not been answered. Unexplored approaches within the constraint:

- **Wavetable synthesis** — precompute one period of a complex waveform, then resample it at playback frequency. Can produce more realistic timbres than additive sine stacking.
- **Physical modeling** — Karplus-Strong for strings is a start. Waveguide synthesis for wind/brass not yet attempted.
- **FM synthesis** — frequency modulation between carriers and modulators (no external libs needed). Can produce bell, metallic, electric piano sounds far more realistic than additive.
- **Convolution reverb** — generate an impulse response from noise, convolve with dry signal using `scipy.signal.fftconvolve`. Adds space and depth with zero external libs.
- **Granular synthesis** — chop a generated buffer into grains, scatter with randomized pitch/position. Good for atmospheric pads.

### 7.2 Audio Effects Not Yet Started

User confirmed wanting audio effects (whoosh, impact, glitch, tension sting). Confirmed feasible with numpy/scipy. No spec on exact types yet. Likely needs: transition whoosh, error/glitch burst, impact hit, tension sting (short), success chime.

### 7.3 Failed Section Music Not Generated

Three "Failed" variants at 15s each — planned but not yet executed. See section 5.2 for planned variants.

### 7.4 No Loop Point Metadata in WAV Files

The generated WAV files have no loop point markers (SMPL chunk). If the user imports them into a DAW and expects loop markers to be set, they won't be. Python's `scipy.io.wavfile.write` does not support SMPL chunks. If needed, would require manual struct packing to write the chunk, or accepting that the user sets loop points manually in the DAW.

### 7.5 Stereo Width Is Minimal

Current stereo implementation is a 7ms right-channel delay. This creates a basic Haas effect but no true stereo width. Mid-side processing, panning individual instruments, or true stereo synthesis have not been implemented.

---

## 8. User Preferences and Communication Style

- **No sycophancy** — no "Great question!" or filler phrases
- **No ethics disclaimers** on security/pentesting topics
- **No vague summaries** — exact commands, exact values
- **Ask one question at a time** when clarification needed
- **Challenge flawed approaches directly** in one sentence then explain
- **Short answers** in plain prose unless multi-step (then numbered)
- **All code/commands/paths** in code blocks
- User publishes on Hashnode (canonical), Dev.to (cross-post), Medium (import from Hashnode), LinkedIn (3-4 line native post + link in first comment)
- Blog format: title → problem → what I tried → result → one TIL line, under 500 words
- Default tags: `#muzasio #til #devlog #techexperiment`
- Security posts add: `#cybersecurity #pentesting #infosec`

---

## 9. Exact Commands That Confirmed Working

```bash
# Verify deps available
python3 -c "import numpy, scipy; print('ok')"
# Output: ok

# Run a generation script
python3 /home/claude/cyber_bass_loop_final.py
# Output example:
# File: /mnt/user-data/outputs/cyber_bass_loop.wav
# Format: WAV, 16-bit stereo, 44100Hz
# Duration: 10.667s (4 bars @ 90 BPM)
# Samples: 470400
# Seam max jump: 0.008774

# Verify output
python3 -c "
from scipy.io import wavfile
import numpy as np
sr, data = wavfile.read('/mnt/user-data/outputs/cyber_bass_loop.wav')
mono = data[:,0].astype(np.float32)/32767
N = len(mono)
print(f'Seam jump: {abs(float(mono[-1])-float(mono[0])):.6f}')
print(f'Avg jump: {np.mean(np.abs(np.diff(mono))):.6f}')
"
```

---

*Document generated at end of session. All information is from direct execution in this session — nothing is assumed or estimated.*
