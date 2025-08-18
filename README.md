# LatentSDF – Auto-Decoder Tutorial & Explorer  

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)    [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)  [![Conda](https://img.shields.io/badge/Conda-environment-green.svg)](https://docs.conda.io/)  [![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)


A minimal, end-to-end workflow for learning a continuous latent space of 2D tower shapes using an **auto-decoder** and **Signed Distance Fields (SDFs)**. Train in a Jupyter notebook, then explore/interpolate interactively with `PathSelect.py`.  

---

## Project Layout
```
LatentSDF/
├── AutoDecoder.ipynb         # Train & export the model
├── PathSelect.py             # Interactive latent-space explorer
├── models/                   # Exported artifacts (created after training)
│   ├── decoder_model.h5
│   ├── latent_codes.npy
│   ├── coords_flat.npy
│   └── data.json
├── output/                   # Generated SDF sequences (created at runtime)
├── environment.yaml          # Conda environment (provided)
└── README.md
```

---

## Setup (Conda, from environment.yaml)
```bash
conda env create -f environment.yaml
conda activate latent-sdf
```
> If your environment has a different name in `environment.yaml`, activate that instead.  

---

## Quick Start

### 1) Train & Export (Notebook)
Open `AutoDecoder.ipynb` and run all cells to:
- learn latent codes and the decoder jointly,
- visualize SDF reconstruction and interpolation,
- export the model files to `models/`.

### 2) Explore Interactively
After exporting, run:
```bash
python PathSelect.py
```
**Controls**  
- Left Click: add path points in latent space  
- Right Click: remove last point  
- Hover: preview SDF at cursor  
- Space: generate SDF sequence  
- S: save sequence to `output/`  
- C: clear path  
- H: toggle help  

---

## How It Works

### Auto‑decoder: trainable latent codes
Each training sample has its own latent vector that is optimized together with the decoder weights.
```python
# num_shapes must match your dataset
LATENT_DIM = 2      # 2D latent space for visualization
COORD_DIM  = 2      # (x, y)
INPUT_DIM  = LATENT_DIM + COORD_DIM
NUM_SHAPES = num_shapes

# Example initialization (Keras/TF)
latent_codes = tf.Variable(
    tf.random.normal([NUM_SHAPES, LATENT_DIM]), name="latent_codes"
)
```

### Decoder network (your current architecture)
The MLP maps concatenated `[latent_code(2), x, y]` → SDF value.
```python
from tensorflow import keras
from tensorflow.keras import layers

# Auto-decoder hyperparameters
LATENT_DIM = 2      # 2D latent space for visualization
COORD_DIM  = 2      # 2D spatial coordinates (x, y)
INPUT_DIM  = LATENT_DIM + COORD_DIM  # Combined input size
NUM_SHAPES = num_shapes

def create_decoder():
    """
    Decoder Network Architecture

    Input: [latent_code (2D) + coordinate (2D)] = 4D vector
    Output: SDF value at that coordinate

    The network learns to decode latent representations into geometry
    """
    decoder = keras.Sequential([
        layers.Input(shape=(INPUT_DIM,)),

        # Deep network to capture complex shape relationships
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),

        # Output single SDF value
        layers.Dense(1, activation='linear')
    ])
    return decoder

# Create decoder network
decoder = create_decoder()
print("Decoder Architecture:")
decoder.summary()
```

### Training loop (conceptual)
You optimize both `decoder` weights and `latent_codes` to fit ground‑truth SDF samples.
```python
optimizer = tf.keras.optimizers.Adam(1e-3)

@tf.function
def train_step(batch_coords, batch_sdf, batch_indices):
    # batch_indices selects the latent code for each sample
    z = tf.gather(latent_codes, batch_indices)                  # [B, LATENT_DIM]
    x = tf.concat([z, batch_coords], axis=-1)                   # [B, LATENT_DIM+2]
    pred = decoder(x)                                           # [B, 1]

    loss = tf.reduce_mean(tf.square(pred - batch_sdf))          # L2 SDF loss
    # Optional: regularize latent norms to keep space well‑behaved
    loss += 1e-4 * tf.reduce_mean(tf.square(z))

    # Compute and apply gradients for both decoder and latent codes
    with tf.GradientTape() as tape:
        pass  # (left minimal on purpose for brevity)
```
> Implement your own batching & gradient updates in the notebook; the key idea is to backprop through both the decoder and the selected latent codes.

### Interpolation in latent space
Linearly mix latent codes to morph shapes; `PathSelect.py` lets you draw a path and export frames.
```python
def lerp(z0, z1, t):
    return (1.0 - t) * z0 + t * z1
```

### Modify layers quickly
Experiment with depth/width or activations to trade off smoothness vs. detail.
```python
# Change widths
W = 256
decoder = keras.Sequential([
    layers.Input(shape=(INPUT_DIM,)),
    layers.Dense(W, activation='relu'),
    layers.Dense(W, activation='relu'),
    layers.Dense(W, activation='relu'),
    layers.Dense(1)
])

# Swap activations (e.g., 'tanh' for smoother fields)
# layers.Dense(128, activation='tanh')

# Add normalization / dropout (optional)
# layers.LayerNormalization(), layers.Dropout(0.1)
```


By drawing a path in latent space, you morph between shapes and export sequences.  

---

## Expected Model Files
Needed in `models/`:
- `decoder_model.h5`  
- `latent_codes.npy`  
- `coords_flat.npy`  
- `data.json`  

---

## Troubleshooting
- TensorFlow/Keras version mismatches may break `h5` loads. Re-exporting usually fixes it.  
- Ensure all four model files are present before launching `PathSelect.py`.  

---

## License
Open source for education and research.  
