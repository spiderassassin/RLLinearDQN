# Helper script for making plots using just the raw data.

import matplotlib
import matplotlib.pyplot as plt
import os
import shutil

IN_FILE = "input.csv"
OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)
# Save images to file.
matplotlib.use('Agg')
LABELS = ["Linear", "Nonlinear"]
COLOURS = ['b', 'r']

# Generate the aggregate graph of the number of layers vs. the average duration taken to reach the max reward.
def gen_aggregate_graph(data, filename, xLabel, yLabel):
    fig = plt.figure(1)
    keys = data.keys()
    values = data.values()
    x = range(len(keys))
    width_offset = 0.4
    # Create multiple bar charts of the data, with the layers as categories.
    for i in range(len(list(values)[0])):
        plt.bar([j + i * width_offset - width_offset / 2 for j in x], [list(values)[j][i] for j in range(len(keys))], width=width_offset, color=COLOURS[i], label=LABELS[i])
    plt.xticks(x, keys)
    plt.legend()

    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    fig.savefig(os.path.join(OUT_DIR, filename))
    plt.close(fig)


if __name__ == '__main__':
    # Clean out the output directory.
    for file in os.listdir(OUT_DIR):
        # By default remove files.
        try:
            os.remove(os.path.join(OUT_DIR, file))
        # Remove directories if present (don't need to be empty).
        except:
            shutil.rmtree(os.path.join(OUT_DIR, file))

    # Read from the input csv file.
    with open(IN_FILE, 'r') as f:
        lines = f.readlines()

        # Extract the average durations.
        avg_durations = {}
        # And the average number of layer passes.
        avg_layer_passes = {}

        for line in lines:
            key = line.split(',')[0]
            # If it's a new entry, create a new entry in the dictionary, with the value as a list.
            if key not in avg_durations:
                avg_durations[key] = []
            avg_durations[key].append(float(line.split(',')[1]))
            # Do the same for the average number of layer passes.
            if key not in avg_layer_passes:
                avg_layer_passes[key] = []
            avg_layer_passes[key].append(float(line.split(',')[2]))

        gen_aggregate_graph(avg_durations, "aggregate_timesteps.png", "Size of Network (Layers)", "Performance (Timesteps)")
        gen_aggregate_graph(avg_layer_passes, "aggregate_layers.png", "Size of Network (Layers)", "Energy (Total Number of Layers Used During Learning)")

