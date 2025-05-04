import os
import re
from pydub import AudioSegment
import numpy as np
import librosa
import whisper

# Configurations
MIN_SILENCE_DURATION = 0.145  # seconds
LONG_SILENCE_THRESHOLD = 0.5  # seconds
OUTPUT_TEXT_FILE = "transcript_.txt"

def transcribe_audio(mp3_path):
    print("Transcribing audio...")
    model = whisper.load_model("small", device="cuda")
    result = model.transcribe(mp3_path, language='en', word_timestamps=True)
    return result

def detect_pauses(mp3_path):
    print("Detecting pauses...")
    audio = AudioSegment.from_mp3(mp3_path)
    wav_path = "temp_for_silence.wav"
    audio.export(wav_path, format="wav")

    y, sr = librosa.load(wav_path, sr=None)
    intervals = librosa.effects.split(y, top_db=30)

    silences = []
    long_silences = []

    for i in range(1, len(intervals)):
        prev_end = intervals[i - 1][1] / sr
        curr_start = intervals[i][0] / sr
        gap_duration = curr_start - prev_end

        if gap_duration >= MIN_SILENCE_DURATION:
            silences.append({
                "start": prev_end,
                "end": curr_start,
                "duration": gap_duration,
                "type": "acoustic"
            })
            if gap_duration >= LONG_SILENCE_THRESHOLD:
                long_silences.append({
                    "start": prev_end,
                    "end": curr_start,
                    "duration": gap_duration,
                    "type": "acoustic"
                })

    os.remove(wav_path)
    return silences, long_silences

def extract_syntactic_pauses(words):
    syntactic_pauses = []
    for i, word_info in enumerate(words):
        word_text = word_info['word'].strip()
        word_end = word_info['end']
        if word_text.endswith('.') or word_text.endswith(','):
            pause_duration = 0.2  # 200ms typical syntactic pause
            syntactic_pauses.append({
                "start": word_end,
                "end": word_end + pause_duration,
                "duration": pause_duration,
                "type": "syntactic",
                "mark": word_text[-1]
            })
    return syntactic_pauses

def insert_silence_markers(words, silences, long_silences):
    markers = []
    for s in silences:
        markers.append({'time': s['start'], 'marker': '/'})
    for ls in long_silences:
        markers.append({'time': ls['start'], 'marker': '//'})

    markers = sorted(markers, key=lambda x: x['time'])

    output = ""
    last_word_end = None
    marker_idx = 0

    for word_info in words:
        word_text = word_info['word']
        word_start = word_info['start']
        word_end = word_info['end']

        if last_word_end is not None:
            while marker_idx < len(markers) and markers[marker_idx]['time'] < word_start:
                output += f" {markers[marker_idx]['marker']}"
                marker_idx += 1

        output += f" {word_text}"
        last_word_end = word_end

    processed = output.strip()

    # Merge multiple / into //
    processed = re.sub(r'(/\s+){2,}', ' //', processed)
    processed = re.sub(r'/\s+//|//\s+/', ' //', processed)
    processed = re.sub(r'([.,])\s+//', r'\1', processed)

    return processed

def calculate_speech_features(words, audio_duration):
    n = len(words)
    m = sum(len(w['word']) for w in words)
    T_s = audio_duration
    T = sum(w['end'] - w['start'] for w in words)

    # Use provided confidences if available; otherwise fallback to 1.0
    likelihoods = [w.get('probability', 1.0) for w in words]

    t_i_list = [w['end'] - w['start'] for w in words]

    R = m / T_s if T_s > 0 else 0

    L1 = sum(likelihoods)
    L2 = L1 / n if n > 0 else 0
    L3 = L1 / m if m > 0 else 0
    L4 = L1 / T if T > 0 else 0
    L5 = sum([l / t for l, t in zip(likelihoods, t_i_list) if t > 0]) / n if n > 0 else 0
    L6 = L4 / R if R > 0 else 0
    L7 = L5 / R if R > 0 else 0

    N_v = 0  # Placeholder for vowel metrics
    S_v = 0
    Sn_v = 0

    return {
        'L1': L1, 'L2': L2, 'L3': L3, 'L4': L4, 'L5': L5, 'L6': L6, 'L7': L7,
        'R': R,
        'S_v': S_v, 'Sn_v': Sn_v,
        'T_s': T_s, 'T': T,
        'n': n, 'm': m
    }

def extract_word_gaps(words):
    word_gaps = []
    for i in range(1, len(words)):
        prev_end = words[i - 1]['end']
        curr_start = words[i]['start']
        gap_duration = curr_start - prev_end
        if gap_duration > 0:
            word_gaps.append(gap_duration)
    return word_gaps

def calculate_rhythm_metrics_word_level(word_gaps,confidence):
    durations = word_gaps

    percentX = np.sum(durations) / len(durations) * 100 if len(durations) > 0 else 0
    stddevX = np.std(durations) if len(durations) > 0 else 0
    varcoX = np.var(durations) * 100 / np.mean(durations) if np.mean(durations) != 0 else 0

    rpviX = np.mean([abs(durations[i + 1] - durations[i]) for i in range(len(durations) - 1)]) if len(durations) > 1 else 0
    npviX = np.mean([
        abs((durations[i + 1] - durations[i]) / ((durations[i + 1] + durations[i]) / 2))
        for i in range(len(durations) - 1)
    ]) if len(durations) > 1 else 0

    return {
        'percentX': percentX,
        'stddevX': stddevX,
        'varcoX': varcoX,
        'rpviX': rpviX,
        'npviX': npviX,
        'avgWconf': np.mean(confidence),
        'stdWconf': np.std(confidence),
        'minWconf': np.min(confidence),
        'maxWconf': np.max(confidence)
    }

def evaluate_speaking(mp3_path):
    result = transcribe_audio(mp3_path)
    words = []
    confidence_list = []
    for segment in result['segments']:
        if 'words' in segment:
            for w in segment['words']:
                # Add confidence if available
                confidence = w.get('probability', segment.get('avg_logprob', 0))
                words.append({
                    'word': w['word'],
                    'start': w['start'],
                    'end': w['end'],
                    'probability': confidence
                })
                confidence_list.append(confidence)


    silences, long_silences = detect_pauses(mp3_path)
    syntactic_pauses = extract_syntactic_pauses(words)

    transcript_with_markers = insert_silence_markers(words, silences, long_silences)

    audio = AudioSegment.from_mp3(mp3_path)
    audio_duration = len(audio) / 1000.0

    features = calculate_speech_features(words, audio_duration)

    word_gaps = extract_word_gaps(words)
    rhythm_metrics = calculate_rhythm_metrics_word_level(word_gaps,np.array(confidence_list))

    report_lines = []
    report_lines.append("Transcript with silence markers:\n")
    report_lines.append(transcript_with_markers + "\n")

    report_lines.append("\n--- Detailed Transcript (word, start, end, confidence) ---\n")
    for word_info in words:
        word = word_info.get('word', '')
        start = word_info.get('start', 0)
        end = word_info.get('end', 0)
        prob = word_info.get('probability', 1.0)
        report_lines.append(f"{word}, {start:.2f}, {end:.2f}, {prob:.3f}\n")

    report_lines.append("\n--- Pronunciation Features (Chen et al.) ---\n")
    pronunciation_features = {
        'L1': ('Summation of likelihoods of all the individual words', features['L1']),
        'L2': ('Average likelihood across all words', features['L2']),
        'L3': ('Average likelihood across all letters', features['L3']),
        'L4': ('Average likelihood per second', features['L4']),
        'L5': ('Average likelihood density across all words', features['L5']),
        'L6': ('L4 normalized by the rate of speech', features['L6']),
        'L7': ('L5 normalized by the rate of speech', features['L7']),
        'R': ('Rate of speech: letters / total duration', features['R']),
        'S_v': ('Average vowel duration deviations', features['S_v']),
        'Sn_v': ('Average normalized vowel duration deviations', features['Sn_v']),
    }

    for k, (description, value) in pronunciation_features.items():
        report_lines.append(f"{k}: {value} - {description}\n")

    report_lines.append("\n--- Speech Metrics (T_s, T, n, m) ---\n")
    report_lines.append("T_s, T, n, m\n")
    report_lines.append(f"{features['T_s']}, {features['T']}, {features['n']}, {features['m']}\n")

    report_lines.append("\n--- Rhythm Metrics (Word-to-Word Gaps) ---\n")
    report_lines.append("Metric, Description, Value\n")
    for metric, value in rhythm_metrics.items():
        report_lines.append(f"{metric}, {value}\n")

    with open(OUTPUT_TEXT_FILE, "w", encoding="utf-8") as f:
        f.writelines(report_lines)

    print("\n".join(report_lines))
    print(f"\nTranscript and features saved to {OUTPUT_TEXT_FILE}")

# Example usage
if __name__ == "__main__":
    mp3_file = "audio.mp3"  # Replace with your actual file
    evaluate_speaking(mp3_file)
