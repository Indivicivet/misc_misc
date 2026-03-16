"""
python data_value_listener_dummy_example.py | python data_value_listener.py
"""

import time
import random
import sys

print("Producer started. Emitting data...")
try:
    while True:
        val_a = random.uniform(0, 10)
        val_b = random.uniform(0, 10)
        print(f"value_a:{val_a:.2f}", flush=True)
        print(f"value_b:{val_b:.2f}", flush=True)
        print(random.choice(["Alice", "Bob"]))
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\nProducer stopped.")
