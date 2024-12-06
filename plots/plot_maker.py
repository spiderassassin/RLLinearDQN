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

# Generate the aggregate graph of the number of layers vs. the average duration taken to reach the max reward.
def gen_aggregate_graph(data, filename, xLabel, yLabel):
    fig = plt.figure(1)
    keys = data.keys()
    values = data.values()
    x = range(len(keys))
    # Create a bar chart of the data, with the layers as categories.
    plt.bar(x, values, color='blue')
    plt.xticks(x, keys)

    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    fig.savefig(os.path.join(OUT_DIR, filename))
    plt.close(fig)
    
def gen_new_graph(data_linear, data_non_linear, filename, xLabel, yLabel):
    import numpy as np
    import pandas as pd
    #fig = plt.figure(1)
    keys = data_linear.keys()
    values = list(data_linear.values())
    nl_values = list(data_non_linear.values())
    x = list(keys)
    
    #calculate means
    lin_means = np.mean(values, axis=1)
    nl_means = np.mean(nl_values, axis=1)
    lin_stdev = np.std(values, axis=1)
    nl_stdev = np.std(nl_values, axis=1)
    
    print(values, nl_values)
    print(lin_means, nl_means)
    print(lin_stdev, nl_stdev)

    # lin_err = np.concatenate((lin_stdev, lin_stdev),axis=0).T
    # nl_err = np.concatenate((nl_stdev, nl_stdev),axis=0).T
    
    # print(lin_err)
    
    # print(np.concatenate((lin_err, nl_err), axis=0))
    
    y_err = pd.DataFrame(np.array([lin_stdev, nl_stdev]).T, index=x, columns=["Linear", "Non-Linear"])
    
    temp = np.array([lin_means, nl_means]).T
    df = pd.DataFrame(data=temp, index=x, columns=["Linear", "Non-Linear"])
    # Create a bar chart of the data, with the layers as categories.
   
    
    # plt.plot(x, x*list(values)[0], 'r--')

    
    
    fig = df.plot(kind="bar", yerr=y_err).get_figure()
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    
    plt.gca().xaxis.set_tick_params(rotation=0)
    fig.savefig(os.path.join(OUT_DIR, filename))
    plt.close(fig)
    
def gen_ablation_graph(data, filename, xLabel, yLabel):
    fig = plt.figure(1)
    
    markers = ['bo', 'b^', 'ro', 'r^', 'ko']
    
    for i in range(len(data)):
        keys = data[i].keys()
        values = data[i].values()
        x = range(len(keys))
        # Create a bar chart of the data, with the layers as categories.
        plt.scatter(x, values, markers[i])
        plt.xticks(x, keys)

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

        # # Extract the average durations.
        # avg_durations = {}
        # # And the average number of layer passes.
        # avg_layer_passes = {}

        # for line in lines:
        #     avg_durations[line.split(',')[0]] = float(line.split(',')[1])
        #     avg_layer_passes[line.split(',')[0]] = float(line.split(',')[2])


        # Changed to add results from 5 seeds from csv
        # First 5 numbers in row are from linear results
        # Second 5 are from non-linear results
        lin = {}
        nl = {}
        import numpy as np
        for line in lines:
            lin[line.split(',')[0]] = np.array(line.split(',')[1:6]).astype(float)
            nl[line.split(',')[0]] = np.array(line.split(',')[6:]).astype(float)


        # gen_aggregate_graph(avg_durations, "aggregate_timesteps.png", "Size of Network (Layers)", "Performance (Timesteps)")
        # gen_aggregate_graph(avg_layer_passes, "aggregate_layers.png", "Size of Network (Layers)", "Energy (Total Number of Layers Used During Learning)")

        gen_new_graph(lin, nl, "resnet_time_graph.png", "Size of Network (Layers)", "Number of FLOPs")