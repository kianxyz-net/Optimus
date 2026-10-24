#!/usr/bin/env python3
"""
Live cognitive-load readout.

    python live.py --board synthetic                      # no hardware
    python live.py --board unicorn --serial-port COM3
    python live.py --board lsl
    python live.py --board synthetic --model decoder.joblib

Flow:
  1. 60 s baseline. The person sits quietly. This fits the per-individual
     normaliser, and it is not optional -- without it the readout is measuring
     who the person is, not what state they are in.
  2. Live loop. Every second, take the last 2 s, featurise, z-score against
     that person's baseline, predict, smooth, print.

Output is P(high load) plus a three-way verdict with an abstention band. If
you remember one thing: the abstention band is a feature. A readout that says
"uncertain" 15% of the time and is right 90% of the rest is far more useful
than one that always answers and is right 75% of the time.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from cogload.features import extract_window, extract_recording, BaselineNormalizer
from cogload.model import decide

BAR_WIDTH = 40


# --------------------------------------------------------------------------
# Stream backends
# --------------------------------------------------------------------------


class BrainFlowStream:
    """BrainFlow backend. Works with Unicorn, Cyton, Muse, Synthetic, etc."""

    BOARD_IDS = {
        "synthetic": -1,
        "unicorn": 8,
        "cyton": 0,
        "muse_2": 38,
        "muse_s": 39,
    }

    def __init__(self, board: str, serial_port: str = "", mac_address: str = ""):
        from brainflow.board_shim import BoardShim, BrainFlowInputParams

        if board not in self.BOARD_IDS:
            raise ValueError(
                f"unknown board {board!r}; known: {sorted(self.BOARD_IDS)}. "
                "For anything else pass the numeric BrainFlow board id."
            )
        self.board_id = self.BOARD_IDS[board]

        params = BrainFlowInputParams()
        params.serial_port = serial_port
        params.mac_address = mac_address

        BoardShim.disable_board_logger()
        self.board = BoardShim(self.board_id, params)
        self.sfreq = float(BoardShim.get_sampling_rate(self.board_id))
        self.eeg_rows = BoardShim.get_eeg_channels(self.board_id)
        try:
            self.ch_names = BoardShim.get_eeg_names(self.board_id).split(",")
        except Exception:
            self.ch_names = [f"ch{i}" for i in range(len(self.eeg_rows))]

    def start(self):
        self.board.prepare_session()
        self.board.start_stream()

    def stop(self):
        try:
            self.board.stop_stream()
            self.board.release_session()
        except Exception:
            pass

    def get_recent(self, n_samples: int) -> np.ndarray | None:
        data = self.board.get_current_board_data(n_samples)
        if data.shape[1] < n_samples:
            return None
        return data[self.eeg_rows, :]


class LSLStream:
    """Lab Streaming Layer backend, for anything that already speaks LSL."""

    def __init__(self, stream_type: str = "EEG", buffer_sec: float = 10.0):
        from pylsl import StreamInlet, resolve_byprop

        print(f"resolving LSL stream of type {stream_type!r} ...")
        streams = resolve_byprop("type", stream_type, timeout=10)
        if not streams:
            raise RuntimeError(f"no LSL stream of type {stream_type!r} found")

        self.inlet = StreamInlet(streams[0], max_buflen=int(buffer_sec))
        info = self.inlet.info()
        self.sfreq = float(info.nominal_srate())
        n_ch = info.channel_count()

        self.ch_names = []
        ch = info.desc().child("channels").child("channel")
        for _ in range(n_ch):
            self.ch_names.append(ch.child_value("label") or f"ch{len(self.ch_names)}")
            ch = ch.next_sibling()

        self._buf = np.zeros((n_ch, int(buffer_sec * self.sfreq)))

    def start(self):
        pass

    def stop(self):
        pass

    def get_recent(self, n_samples: int) -> np.ndarray | None:
        chunk, _ = self.inlet.pull_chunk(timeout=0.0)
        if chunk:
            new = np.asarray(chunk).T
            k = new.shape[1]
            if k >= self._buf.shape[1]:
                self._buf = new[:, -self._buf.shape[1]:]
            else:
                self._buf = np.roll(self._buf, -k, axis=1)
                self._buf[:, -k:] = new
        return self._buf[:, -n_samples:]


# --------------------------------------------------------------------------


def collect(stream, seconds: float, win_samples: int, label: str) -> np.ndarray:
    """Collect `seconds` of data, returning (n_channels, n_samples)."""
    print(f"\n{label}")
    chunks = []
    t0 = time.time()
    while time.time() - t0 < seconds:
        time.sleep(0.5)
        buf = stream.get_recent(win_samples)
        if buf is not None:
            chunks.append(buf)
        elapsed = time.time() - t0
        bar = int(BAR_WIDTH * elapsed / seconds)
        print(f"\r  [{'#' * bar}{'.' * (BAR_WIDTH - bar)}] {elapsed:4.0f}/{seconds:.0f}s",
              end="", flush=True)
    print()
    if not chunks:
        raise RuntimeError("no data received from the stream")
    return np.concatenate(chunks, axis=1)


TAGS = {"high load": "HIGH", "low load": " LOW", "uncertain": "  ??",
        "rest": "REST", "task": "TASK"}


def render(p: float, verdict: str, raw: float, conf: float = float("nan"),
           n_levels: int = 2) -> str:
    filled = int(BAR_WIDTH * min(max(p, 0.0), 1.0))
    bar = "#" * filled + "." * (BAR_WIDTH - filled)
    tag = TAGS.get(verdict, verdict[:4].upper())

    if n_levels > 2:
        level = p * (n_levels - 1)
        meter = f"lvl {level:4.2f}/{n_levels - 1}"
    else:
        meter = f"{p:5.1%}"

    conf_s = "" if np.isnan(conf) else f"  conf {conf:4.0%}"
    return f"\r  [{bar}] {meter}  {tag}{conf_s}   (raw {raw:5.1%})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="synthetic",
                    help="synthetic | unicorn | cyton | muse_2 | muse_s | lsl")
    ap.add_argument("--serial-port", default="")
    ap.add_argument("--mac-address", default="")
    ap.add_argument("--model", default="decoder.joblib",
                    help="trained model; omit for unsupervised distance mode")
    ap.add_argument("--baseline-sec", type=float, default=60.0)
    ap.add_argument("--win", type=float, default=2.0)
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.3, help="smoothing factor")
    ap.add_argument("--low", type=float, default=0.35)
    ap.add_argument("--high", type=float, default=0.65)
    ap.add_argument("--min-conf", type=float, default=0.5,
                    help="ordinal models: below this, report 'uncertain'")
    ap.add_argument("--log", default="", help="append readings to this CSV")
    args = ap.parse_args()

    if args.board == "lsl":
        stream = LSLStream()
    else:
        stream = BrainFlowStream(args.board, args.serial_port, args.mac_address)

    print(f"stream: {args.board}  {stream.sfreq:.0f} Hz  "
          f"{len(stream.ch_names)} ch  {stream.ch_names}")

    win_samples = int(args.win * stream.sfreq)
    stream.start()
    time.sleep(2.0)

    try:
        # ---- baseline -------------------------------------------------
        base_raw = collect(
            stream, args.baseline_sec, win_samples,
            f"BASELINE -- sit still, eyes open, relax for {args.baseline_sec:.0f}s",
        )
        Xb, names = extract_recording(
            base_raw, stream.sfreq, stream.ch_names,
            species="human", win_sec=args.win, step_sec=args.step,
        )
        norm = BaselineNormalizer().fit(Xb)
        print(f"  baseline fitted on {len(Xb)} windows")

        # ---- model ----------------------------------------------------
        decoder = None
        n_levels = 2
        if args.model:
            try:
                import joblib
                bundle = joblib.load(args.model)
                trained_names = bundle.get("feature_names")
                n_levels = int(bundle.get("n_levels", 2))
                if trained_names != names:
                    # Hard stop rather than a warning. A montage mismatch means
                    # feature k at predict time is not feature k at train time,
                    # so the model will emit confident, meaningless numbers. The
                    # unsupervised fallback below is wrong-but-honest; a
                    # misaligned classifier is wrong-and-convincing.
                    missing = set(trained_names or []) - set(names)
                    extra = set(names) - set(trained_names or [])
                    print("  MONTAGE MISMATCH -- not loading the model.")
                    print(f"    trained on {len(trained_names or [])} features, "
                          f"this stream gives {len(names)}")
                    if missing:
                        print(f"    missing: {sorted(missing)[:6]}")
                    if extra:
                        print(f"    unexpected: {sorted(extra)[:6]}")
                    print("    Usually means the channel labels are generic "
                          "(ch0, ch1...) instead of 10-20 names.")
                    print("    Falling back to unsupervised distance mode.")
                else:
                    decoder = bundle["decoder"]
                    kind = "ordinal" if n_levels > 2 else "binary"
                    print(f"  loaded {kind} model {args.model} "
                          f"({n_levels} levels)")
            except FileNotFoundError:
                print(f"  no model at {args.model}; "
                      "falling back to unsupervised distance mode")
            except Exception as exc:
                print(f"  could not load model ({exc}); using distance mode")

        logfh = open(args.log, "a") if args.log else None
        if logfh:
            logfh.write("timestamp,p_load,p_smoothed,verdict,confidence\n")

        # ---- live loop ------------------------------------------------
        print("\nLIVE  (ctrl-C to stop)\n")
        smoothed = None
        while True:
            buf = stream.get_recent(win_samples)
            if buf is None:
                time.sleep(args.step)
                continue

            f = extract_window(buf, stream.sfreq, stream.ch_names, species="human")
            z = norm.transform(np.array([[f[k] for k in names]]))

            if decoder is not None and n_levels > 2:
                # Ordinal model: the meter is the expected level, rescaled to
                # 0-1 for display. Verdict and confidence come from the
                # per-level probabilities, not from the rescaled scalar --
                # those are two different quantities and conflating them is
                # how you end up reporting 50% confidence on a mid-range
                # reading that the model is actually sure about.
                probs = decoder.predict_proba(z)[0]
                level = float(decoder.expected_level(z)[0])
                p = level / (n_levels - 1)
                conf = float(probs.max())
                best = int(np.argmax(probs))
            elif decoder is not None:
                p = float(decoder.predict_load(z)[0])
                conf = max(p, 1.0 - p)
                best = int(p >= 0.5)
            else:
                # No model: report how far this window sits from the person's
                # own baseline, squashed into 0-1. This is a DISTANCE, not a
                # probability -- it says "unusual for you", not "overloaded".
                d = float(np.mean(np.abs(z)))
                p = float(1.0 - np.exp(-d / 2.0))
                conf = float("nan")
                best = None

            smoothed = p if smoothed is None else args.alpha * p + (1 - args.alpha) * smoothed

            if n_levels > 2 and best is not None:
                from cogload.protocol import LEVEL_NAMES
                verdict = (LEVEL_NAMES.get(best, str(best))
                           if conf >= args.min_conf else "uncertain")
            else:
                verdict = decide(smoothed, args.low, args.high)

            print(render(smoothed, verdict, p, conf, n_levels), end="", flush=True)

            if logfh:
                logfh.write(f"{time.time():.3f},{p:.4f},{smoothed:.4f},"
                            f"{verdict},{conf:.4f}\n")
                logfh.flush()

            time.sleep(args.step)

    except KeyboardInterrupt:
        print("\n\nstopped")
    finally:
        stream.stop()
        try:
            logfh.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
