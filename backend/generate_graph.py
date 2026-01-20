# utils/generate_graph.py
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib
matplotlib.use("Agg")  


def csv_to_image(csv_path, output_dir):
    df = pd.read_csv(csv_path)
    filename = os.path.splitext(os.path.basename(csv_path))[0]
    output_path = os.path.join(output_dir, f"{filename}.png")

    # On suppose que les colonnes sont des signaux temporels
    df.plot()
    plt.title(filename)
    plt.savefig(output_path)
    plt.close()

    return output_path
