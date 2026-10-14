"""Coffee farm sensor demo: cleaning, Isolation Forest training and JSON API.

Uses only Python's standard library so the prototype can run without installing
packages. Run with: python server.py
"""
from __future__ import annotations

import json
import math
import random
import statistics
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
FEATURES = ("temperature", "air_humidity", "soil_moisture", "light")
N = 7 * 24 * 4
SEED = 20261013


def raw_data(n=N, seed=SEED):
    rng = random.Random(seed)
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    rows, labels = [], []
    for i in range(n):
        t = i / 96
        temp = 22.5 + 3.1 * math.sin(t * math.tau - 1.1) + rng.gauss(0, .55)
        humidity = 69 - 8 * math.sin(t * math.tau - 1.1) + rng.gauss(0, 2.2)
        soil = 51 - .018 * i + 1.1 * math.sin(t * math.tau / 3) + rng.gauss(0, .65)
        light = max(0, 17 + 13 * max(0, math.sin(t * math.tau - 1.1)) + rng.gauss(0, 2.5))
        label = 0
        # Known synthetic events, with labels retained only for evaluation.
        if 250 <= i < 260:  # isolated heat spike
            temp += 9
            label = 1
        if 380 <= i < 410:  # gradual soil drying
            soil -= (i - 379) * .42
            label = 1
        if 540 <= i < 547:  # correlated hot, dry air and soil
            temp += 6
            humidity -= 19
            soil -= 12
            label = 1
        if rng.random() < .004 and not label:
            light += rng.choice([-1, 1]) * 22  # occasional sensor spike
        stamp = now - timedelta(minutes=15 * (n - 1 - i))
        rows.append({"timestamp": stamp.isoformat(), "temperature": temp,
                     "air_humidity": humidity, "soil_moisture": soil,
                     "light": light})
        labels.append(label)
    # Simulate missing readings and repair by linear interpolation.
    for key in FEATURES:
        for i in (87, 304, 601):
            rows[i][key] = None
        valid = [i for i, row in enumerate(rows) if row[key] is not None]
        for i, row in enumerate(rows):
            if row[key] is None:
                left = max((j for j in valid if j < i), default=valid[0])
                right = min((j for j in valid if j > i), default=valid[-1])
                frac = (i - left) / max(1, right - left)
                row[key] = rows[left][key] * (1 - frac) + rows[right][key] * frac
    # Hampel-style rolling median filter for isolated sensor spikes.
    filtered_spikes = 0
    for key in FEATURES:
        original = [row[key] for row in rows]
        for i in range(3, n - 3):
            window = original[i - 3:i + 4]
            med = statistics.median(window)
            mad = statistics.median(abs(x - med) for x in window)
            if mad > 0 and abs(original[i] - med) > 4.5 * 1.4826 * mad:
                rows[i][key] = med
                filtered_spikes += 1
    return rows, labels, {"missing_filled": 12, "spikes_filtered": filtered_spikes}


def window_features(rows, window=16):
    out = []
    for i, row in enumerate(rows):
        vals = [row[k] for k in FEATURES]
        feat = list(vals)
        start = max(0, i - window + 1)
        for key in FEATURES:
            seq = [rows[j][key] for j in range(start, i + 1)]
            mean = sum(seq) / len(seq)
            sd = math.sqrt(sum((v - mean) ** 2 for v in seq) / len(seq))
            span = max(seq) - min(seq)
            slope = (seq[-1] - seq[0]) / max(1, len(seq) - 1)
            feat.extend((mean, sd, span, slope))
        out.append(feat)
    return out


def normalize(matrix, reference=None):
    reference = reference or matrix
    means = [sum(row[j] for row in reference) / len(reference) for j in range(len(matrix[0]))]
    scales = [math.sqrt(sum((row[j] - means[j]) ** 2 for row in reference) / len(reference)) or 1
              for j in range(len(means))]
    return [[(v - means[j]) / scales[j] for j, v in enumerate(row)] for row in matrix]


def c_factor(n):
    if n <= 1:
        return 0
    if n == 2:
        return 1
    return 2 * (math.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n


def build_tree(data, depth, limit, rng):
    if depth >= limit or len(data) < 2:
        return (None, None, None, len(data))
    dims = len(data[0])
    candidates = [j for j in range(dims) if min(r[j] for r in data) < max(r[j] for r in data)]
    if not candidates:
        return (None, None, None, len(data))
    feat = rng.choice(candidates)
    low, high = min(r[feat] for r in data), max(r[feat] for r in data)
    split = rng.uniform(low, high)
    left = [r for r in data if r[feat] < split]
    right = [r for r in data if r[feat] >= split]
    if not left or not right:
        return (None, None, None, len(data))
    return (feat, split, build_tree(left, depth + 1, limit, rng),
            build_tree(right, depth + 1, limit, rng))


def path_length(row, node, depth=0):
    feat, split, left, right = node
    if feat is None:
        return depth + c_factor(node[3])
    return path_length(row, left if row[feat] < split else right, depth + 1)


def isolation_scores(matrix, seed=SEED, trees=72, sample_size=192, training_fraction=.30):
    # Fit against the initial baseline period, then score the full timeline.
    # Synthetic event labels are never used to fit the forest or choose flags.
    training_count = max(2, int(len(matrix) * training_fraction))
    data = normalize(matrix, matrix[:training_count])
    training = data[:training_count]
    rng = random.Random(seed)
    size = min(sample_size, len(training))
    limit = math.ceil(math.log2(size))
    forest = [build_tree(rng.sample(training, size), 0, limit, rng) for _ in range(trees)]
    norm = c_factor(size)
    return [2 ** (-sum(path_length(row, tree) for tree in forest) / trees / norm)
            for row in data]


def iqr_flags(rows):
    flags = []
    bounds = {}
    for key in FEATURES:
        vals = sorted(row[key] for row in rows)
        q1, q3 = vals[int(.25 * (len(vals) - 1))], vals[int(.75 * (len(vals) - 1))]
        spread = q3 - q1
        bounds[key] = (q1 - 1.5 * spread, q3 + 1.5 * spread)
    for row in rows:
        flags.append(int(any(row[k] < bounds[k][0] or row[k] > bounds[k][1] for k in FEATURES)))
    return flags


def evaluate(flags, labels):
    tp = sum(p and y for p, y in zip(flags, labels))
    fp = sum(p and not y for p, y in zip(flags, labels))
    fn = sum(not p and y for p, y in zip(flags, labels))
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    return {"precision": round(precision, 3), "recall": round(recall, 3),
            "f1": round(2 * precision * recall / (precision + recall), 3) if precision + recall else 0,
            "true_positives": tp, "false_positives": fp, "false_negatives": fn}


def threshold_flags(scores, fraction=.055):
    cutoff = sorted(scores)[int((1 - fraction) * (len(scores) - 1))]
    return [int(s >= cutoff) for s in scores], cutoff


def run_analysis():
    rows, labels, cleaning = raw_data()
    base_flags = iqr_flags(rows)
    raw_scores = isolation_scores([[r[k] for k in FEATURES] for r in rows])
    win_scores = isolation_scores(window_features(rows), seed=SEED + 1)
    if_flags, if_cut = threshold_flags(raw_scores)
    win_flags, win_cut = threshold_flags(win_scores)
    methods = {
        "IQR": {**evaluate(base_flags, labels), "flags": base_flags},
        "Isolation Forest": {**evaluate(if_flags, labels), "flags": if_flags, "cutoff": round(if_cut, 4)},
        "IF + 时间窗口": {**evaluate(win_flags, labels), "flags": win_flags, "cutoff": round(win_cut, 4)},
    }
    latest = rows[-1]
    alert = bool(win_flags[-1]) or latest["soil_moisture"] < 35
    # Return data points at 15-minute resolution for charts and investigation.
    points = [{"timestamp": row["timestamp"], **{k: round(row[k], 2) for k in FEATURES},
               "anomaly": bool(win_flags[i]), "label": bool(labels[i]),
               "score": round(win_scores[i], 4)} for i, row in enumerate(rows)]
    return {"model": "Isolation Forest + 4h rolling features", "window_samples": 16,
            "training_baseline_fraction": .30,
            "samples": len(rows), "labeled_anomalies": sum(labels),
            "cleaning": {**cleaning, "interval_minutes": 15},
            "metrics": {name: {k: v for k, v in result.items() if k != "flags"}
                        for name, result in methods.items()},
            "latest": {k: round(v, 2) for k, v in latest.items() if k != "timestamp"} | {"timestamp": latest["timestamp"]},
            "alerts": [{"level": "high", "metric": "土壤湿度", "value": latest["soil_moisture"],
                        "message": "土壤湿度偏低，请检查滴灌并安排补水。"}] if alert else [],
            "points": points}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/" or path == "/index.html":
            body, kind = (ROOT / "index.html").read_bytes(), "text/html; charset=utf-8"
        elif path == "/api/health":
            body, kind = json.dumps({"status": "ok", "model": "Isolation Forest"}).encode(), "application/json"
        elif path in ("/api/analysis", "/api/evaluate"):
            body, kind = json.dumps(run_analysis(), ensure_ascii=False).encode(), "application/json; charset=utf-8"
        else:
            body, kind = json.dumps({"error": "not found"}).encode(), "application/json"
            self.send_response(404)
            self.send_header("Content-Type", kind)
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        print("%s - %s" % (self.log_date_time_string(), fmt % args))


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 8000), Handler)
    print("Coffee monitor API: http://127.0.0.1:8000  (Ctrl+C to stop)")
    server.serve_forever()
