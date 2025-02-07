# This script reads metrics.csv files and plots four graphs using Matplotlib

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import os
from threading import Thread
import time

plt.style.use("dark_background")
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title

episodic_data = None
AGENT_TRAIN_METRIC_PATH = "models/DQN_mountaincar_metrics.csv"

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Training Metrics")
ax1, ax2, ax3, ax4 = axes.flatten()


# Reading our metric on a different thread
def read_data():
    global episodic_data
    while True:
        try:
            episodic_data = pd.read_csv(AGENT_TRAIN_METRIC_PATH)
        except Exception as e:
            print(e)
        time.sleep(10)


t = Thread(target=read_data, daemon=True)
t.start()


# Functions to plot 4 graphs
def update_plots(i):
    if episodic_data is None:
        return

    try:
        # Total Reward Plot
        ax1.clear()
        ax1.plot(episodic_data["episode"], episodic_data['total_reward'], linewidth=1, color="#2AD4FF")
        ax1.set_title("Total Reward per Episode")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Total Reward")

        # Average Q-Value Plot
        ax2.clear()
        ax2.plot(episodic_data["episode"], episodic_data['avg_q_value'], linewidth=1, color="#FF00FF")
        ax2.set_title("Average Q Value")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Q Value")

        # Episode Length Plot
        ax3.clear()
        ax3.plot(episodic_data["episode"], episodic_data['episode_length'], linewidth=1, color="#00FF00")
        ax3.set_title("Length of Episode")
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Length")

        # Exploration (Epsilon) Plot
        ax4.clear()
        ax4.plot(episodic_data["episode"], episodic_data['epsilon'], linewidth=1, color="#FF6600")
        ax4.set_title("Exploration Rate")
        ax4.set_xlabel("Episode")
        ax4.set_ylabel("Epsilon")

        # Adjust layout for better spacing
        plt.tight_layout(rect=[0, 0, 1, 0.96])

    except Exception as e:
        print(e)


# Update the figure every 10 seconds
anim = FuncAnimation(fig, update_plots, interval=10000)
plt.show()