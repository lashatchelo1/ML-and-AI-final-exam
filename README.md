# Task3
# Lasha-Giorgi Tchelidze 

## Network Traffic Anomaly Detection (Machine Learning Model)

###Overview

This project provides a comprehensive, ready-to-train machine learning pipeline that can distinguish between normal and abnormal network traffic using numerical flow-based features.
It can be used for:

-Intrusion detection

-Network anomaly detection

-Threat hunting and traffic profiling

-Research or lab simulations

By default, the script generates synthetic traffic data, but you can seamlessly replace it with real NetFlow, firewall, or IDS logs.

### How the Model Works

The model is built to classify traffic as:

- 0 – Normal traffic

- 1 – Abnormal (potential attack)

It uses:

StandardScaler to normalize numerical feature distributions
RandomForestClassifier to capture nonlinear patterns in traffic behavior ROC, precision, recall, F1, and confusion matrices to evaluate detection capability

### Project Structure
.
??? README.md        # Documentation
??? Task3.py # Main script for training & evaluation

#### Features & Capabilities

####Synthetic dataset generation

Creates a realistic simulation of normal vs abnormal traffic distributions.

#### Easy switch to real datasets

Replace just 3 lines to load your real CSV with NetFlow-like features.

#### Automatically evaluates using:

Confusion Matrix

Accuracy, Precision, Recall, F1

ROC-AUC

ROC Curve Visualization

#### Supports real-world deployment

You can call pipeline.predict() on live flows or exported logs.

### Example Features

Your dataset can include numeric features like:

-Duration of connection
-Bytes sent / received
-Packets in / out
-Source/Destination port encodings
-Protocol type
-Flow rate
-TCP flags
These features reflect typical NetFlow or firewall log attributes.

### Install required libraries:

pip install numpy pandas scikit-learn matplotlib

### Running the Model

Run the script:
python task3.py

You will see:
-Dataset preview
-Model training
-Metrics
-ROC Curve plot
-Predictions on example flows

### Using Your Own Dataset (Highly Recommended)

Replace the synthetic dataset block with:

df = pd.read_csv("your_network_data.csv")

feature_names = [
    "duration",
    "src_bytes",
    "dst_bytes",
    "packets_in",
    "packets_out",
    "src_port_enc",
    "dst_port_enc",
    "protocol_enc",
    "tcp_flags_enc",
    "flow_rate"
]

X = df[feature_names].values
y = df["label"].values   # must be 0 or 1

Your CSV must contain:

-The same feature columns
-A label column (0 = normal, 1 = abnormal)
If your log fields differ, modify feature_names accordingly.

### Understanding the Output

#### Classification Report

Shows how well the model detects normal vs abnormal traffic.

#### Confusion Matrix

True Positive ? attack correctly detected
False Negative ? dangerous misses
True Negative ? normal correctly ignored
False Positive ? false alert

#### ROC-AUC Score

Measures detection quality even with imbalanced data.

#### ROC Curve Plot

Higher curve ? better anomaly detection.
