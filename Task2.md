# Task2
# Lasha-Giorgi Tchelidze 

## Comprehensive Description of Autoencoder Neural Networks

An autoencoder (AE) is an unsupervised neural network that learns to compress (encode) input data into a lower-dimensional latent representation and then reconstruct (decode) the original input from this compressed form as accurately as possible. Its training objective is to minimize reconstruction error (typically MSE or binary cross-entropy), forcing the network to learn the most salient features of the data distribution.

### Architecture
A basic autoencoder consists of two symmetrical parts:
Encoder: Maps high-dimensional input x ? ?? to a low-dimensional latent vector (bottleneck) z ? ?? (k ? n)
z = f(Wx + b)  (often with non-linear activation like ReLU or tanh)
Decoder: Reconstructs the input from the latent code
x? = g(W'z + b')

The full network is trained end-to-end with loss ?(x, x?), usually Mean Squared Error (MSE) or Binary Cross-Entropy.

###Key Variants

Denoising Autoencoder (DAE): Input is corrupted (e.g., Gaussian noise, masking); forces learning of robust features.
Variational Autoencoder (VAE): Probabilistic encoder outputs ? and ?; latent space is regularized to follow a Gaussian prior ? generative capabilities.
Sparse Autoencoder: Adds sparsity penalty (e.g., KL divergence on hidden activations) to encourage feature selectivity.
Convolutional/Recurrent Autoencoders: Use CNN or RNN layers for images or sequences.
Contractive Autoencoder: Adds Jacobian penalty to make encoding robust to small input perturbations.

Autoencoders learn a compressed identity function, making them powerful for dimensionality reduction, feature extraction, denoising, and anomaly detection.

## Applications in Cybersecurity (with Real-World Impact)
Autoencoders are exceptionally effective in cybersecurity because most attacks are rare anomalies against a background of normal behavior. The reconstruction error becomes a natural anomaly score.
###Network Intrusion Detection (NIDS)

Train an autoencoder on benign traffic only (e.g., NSL-KDD, CIC-IDS2017, or real enterprise NetFlow logs).
Normal packets ? low reconstruction error.
Attack packets (DDoS, port scans, exploits) ? high reconstruction error ? flagged.
Real deployments: DeepAutoIDPS systems at companies like Darktrace and Vectra AI heavily rely on autoencoder-like architectures.

###Malware Traffic Detection

Use sequence autoencoders (LSTM/Transformer-based) on HTTP/S or DNS request sequences.
Benign user browsing patterns reconstruct well; C2 beaconing, DGA domains, or exfiltration patterns fail ? high error.
Example: Kitsune (2018) used an ensemble of autoencoders on network packet features and achieved state-of-the-art unsupervised detection on real IoT networks.

###Fraud and Insider Threat Detection

Build user behavior profiles from system call sequences, API logs, or keystroke dynamics.
Variational autoencoders can model normal employee behavior in enterprise environments (e.g., CERT Insider Threat dataset).
Sudden deviation (data exfiltration, privilege escalation) triggers alerts.

### File and Binary Malware Classification

Train on raw bytes or opcodes of benign executables.
Malware samples deviate in latent space or have high reconstruction loss.
Used in systems like Microsoft’s Windows Defender ML components.

###Log Anomaly Detection

Autoencoders on parsed syslog, Windows Event Logs, or cloud audit logs (AWS CloudTrail).
Rare events (brute-force, lateral movement) produce outliers in reconstruction.

### Industrial Control Systems (ICS/SCADA) Anomaly Detection

Autoencoders on sensor readings (voltage, pressure, temperature) in power grids or water treatment plants.
Physical attacks (e.g., Stuxnet-like manipulation) cause measurable reconstruction spikes.

## Scaled Dot-Product Attention – How It Really Works (with Visualization)

Input tokens ? Embedding + Positional Encoding
          ?
   [Query Q | Key K | Value V]   ? linear projections of the same input
          ?
   Attention Scores = (Q × K?) / ?d_k
          ?
   Attention Weights = softmax(scores)   ? this is what we visualize!
          ?
   Output = Attention Weights × V

##  Visualization of a Real Attention Matrix (8×8 example)
 

 
