"""
python your_script.py | python value_listener.py "current value"
"""

import sys
import re
import argparse

import matplotlib.pyplot as plt


def listen(keywords, max_samples):
    print("Listener started. Waiting for data...")
    plt.ion()  # interactive mode
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Live Data Stream")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Value")
    ax.grid(True)
    datas = {k: [] for k in keywords}
    lines = {k: ax.plot([], [], label=k)[0] for k in keywords}
    ax.legend()
    find_vals = "|".join(keywords).lower()
    # in regex allow spaces after val and separation by : or =
    pattern = re.compile(rf"({find_vals})\s*[:=]\s*([-+]?\d*\.?\d+)")
    try:
        for line in sys.stdin:
            print(line, end="", flush=True)
            matches = pattern.findall(line.lower())
            for key, val in matches:
                val = float(val)
                datas[key].append(val)
                lines[key].set_data(range(len(datas[key])), datas[key])
                num_points = max(len(datas[key]) for key in datas)
                if num_points > max_samples:
                    ax.set_xlim(num_points - max_samples, num_points)
                else:
                    ax.set_xlim(0, max(max_samples, num_points))
                # Dynamically scale the y-axis
                ax.relim()
                ax.autoscale_view(scalex=False, scaley=True)
            # Pause to update the figure (this also processes GUI events)
            plt.pause(0.01)
    except KeyboardInterrupt:
        print("\nListener stopped.")
    finally:
        # Keep the plot open after the stream ends
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "keywords", nargs="*", default=["value_a", "value_b"]
    )
    parser.add_argument(
        "--max_samples", "-m", type=int, default=100
    )
    args = parser.parse_args()
    listen(keywords=args.keywords, max_samples=args.max_samples)
