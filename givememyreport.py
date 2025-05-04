import matplotlib.pyplot as plt
import numpy as np
import re

# --- Configurable ---
FILES = {
    "Your Response": {
        "path": "transcript_myaudio.txt",
        "color": "blue",
        "alpha": 1.0
    },
    "Sample 1": {
        "path": "transcript_sample1.txt",
        "color": "red",
        "alpha": 0.6  # 'a' in your spec
    },
    "Sample 2": {
        "path": "transcript_sample2.txt",
        "color": "red",
        "alpha": 0.3  # 'b' in your spec
    }
}

def count_silences(transcript_text):
    num_silence = len(re.findall(r'\s/\s', transcript_text))
    num_long_silence = len(re.findall(r'\s//\s', transcript_text))
    return num_silence, num_long_silence

def parse_report(filepath):
    with open(filepath, encoding="utf-8") as f:
        content = f.read()

    # --- Count silences ---
    transcript_match = re.search(r'Transcript with silence markers:\n(.*?)---', content, re.S)
    transcript_text = transcript_match.group(1) if transcript_match else ""
    num_silence, num_long_silence = count_silences(transcript_text)

    # --- Pronunciation Features ---
    pronun_section = re.search(r'--- Pronunciation Features.*?---\s*(.*?)---', content, re.S)

    pronun_data = {}
    if pronun_section:
        for line in pronun_section.group(1).splitlines():
            match = re.match(r'([A-Za-z0-9_]+): ([\d\.eE+-]+)', line.strip())
            if match:
                key = match.group(1)
                val = float(match.group(2))
                pronun_data[key] = val

    # --- Speech Metrics ---
    speech_metrics_match = re.search(r'T_s, T, n, m\n(.*?)\n', content)
    speech_data = {}
    if speech_metrics_match:
        vals = [float(x.strip()) for x in speech_metrics_match.group(1).split(",")]
        speech_data = dict(zip(['T_s', 'T', 'n', 'm'], vals))
        speech_data['/'] = num_silence
        speech_data['//'] = num_long_silence

    # --- Rhythm Metrics ---
    rhythm_section = re.search(r'--- Rhythm Metrics.*?---\nMetric, Description, Value\n(.*)', content, re.S)
    rhythm_data = {}
    if rhythm_section:
        for line in rhythm_section.group(1).splitlines():
            parts = line.strip().split(',')
            if len(parts) == 2:
                key, val = parts
                try:
                    rhythm_data[key.strip()] = float(val.strip())
                except ValueError:
                    pass

    return pronun_data, speech_data, rhythm_data

# --- Extract all data ---
all_pronun = {}
all_speech = {}
all_rhythm = {}

for label, cfg in FILES.items():
    pronun, speech, rhythm = parse_report(cfg['path'])
    all_pronun[label] = pronun
    all_speech[label] = speech
    all_rhythm[label] = rhythm
    print(f"{label} Pronunciation Data: {pronun}")
    print(f"{label} Speech Data: {speech}")
    print(f"{label} Rhythm Data: {rhythm}")
    print("=" * 40)

# --- Improved Subplot Function ---

def plot_metric_subplots(metric_dict, title, ylabel, keys):
    num_metrics = len(keys)
    nrows = 2
    ncols = (num_metrics + 1) // 2  # Ensures ceil division for odd numbers

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 5 * nrows), sharey=False)

    # Flatten axes to 1D list for easy looping
    axes = axes.flatten()

    for idx, key in enumerate(keys):
        ax = axes[idx]
        bar_vals = []
        bar_colors = []
        bar_labels = []
        bar_alphas = []

        for label, cfg in FILES.items():
            val = metric_dict[label].get(key, 0)
            bar_vals.append(val)
            bar_colors.append(cfg['color'])
            bar_labels.append(label)
            bar_alphas.append(cfg['alpha'])

        bars = ax.bar(bar_labels, bar_vals, color=bar_colors)
        for bar, alpha_val in zip(bars, bar_alphas):
            bar.set_alpha(alpha_val)

        ax.set_title(key)
        ax.set_xlabel('Source')
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(bar_labels)))
        ax.set_xticklabels(bar_labels, rotation=15)

    # Hide unused subplots (in case of odd number of keys)
    for j in range(len(keys), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    

# Print meanings for each metric
def print_metric_meanings():
    # Pronunciation Features
    print("Pronunciation Features:")
    print("L1: Likelihood of the entire transcript (Higher values suggest better clarity of speech).")
    print("L2: Average likelihood across all words (Higher values indicate more natural speech).")
    print("L3: Average likelihood across all letters (Represents how likely each letter is).")
    print("L4: Average likelihood per second (How confident the model is in speech over time).")
    print("L5: Average likelihood density across all words (Density of likelihoods).")
    print("L6: L4 normalized by rate of speech (Time-based adjustment of L4).")
    print("L7: L5 normalized by rate of speech (Time-based adjustment of L5).")
    print("R: Rate of speech: Letters per total duration (The speed of speech).")
    print("S_v: Average vowel duration deviations (Indicates vowel articulation).")
    print("Sn_v: Average normalized vowel duration deviations (Standardized vowel articulation).")
    
    # Speech Metrics
    print("\nSpeech Metrics:")
    print("T_s: Total speech time (Total time spent speaking).")
    print("T: Total time (Total time including speech and pauses).")
    print("n: Total number of words (Speech word count).")
    print("m: Total number of syllables (Total syllables in speech).")
    print("/: Number of short pauses (Quick pauses between words).")
    print("//: Number of long pauses (Longer pauses between sentences).")
    
    # Rhythm Metrics
    print("\nRhythm Metrics:")
    print("percentX: Percentage of speech with a rhythmic pattern.")
    print("stddevX: Standard deviation of rhythmic patterns (Variability in rhythm).")
    print("varcoX: Variance of the rhythm (How much the rhythm fluctuates).")
    print("rpviX: Rhythmic periodic variability index (Rhythmic consistency over time).")
    print("npviX: Non-periodic variability index (Consistency of non-rhythmic patterns).")


# 1️⃣ Pronunciation Features Plot
pronun_keys = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'R', 'S_v', 'Sn_v']
plot_metric_subplots(all_pronun, 'Pronunciation Features Comparison', 'Value', pronun_keys)

# 2️⃣ Speech Metrics Plot (+ / and //)
speech_keys = ['T_s', 'T', 'n', 'm', '/', '//']
plot_metric_subplots(all_speech, 'Speech Metrics Comparison', 'Value', speech_keys)

# 3️⃣ Rhythm Metrics Plot
rhythm_keys = ['percentX', 'stddevX', 'varcoX', 'rpviX', 'npviX','avgWconf','stdWconf','minWconf','maxWconf']
plot_metric_subplots(all_rhythm, 'Rhythm Metrics Comparison', 'Value', rhythm_keys)

print_metric_meanings()

plt.show()