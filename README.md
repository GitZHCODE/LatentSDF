# Latent Space Exploration for SDF Tower Shapes

A complete tutorial and interactive application for learning continuous latent space representations of 2D tower shapes using auto-decoder neural networks and Signed Distance Fields (SDFs).

![Tower Shapes](https://img.shields.io/badge/Project-Latent%20SDF-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange) ![Python](https://img.shields.io/badge/Python-3.7+-green)

## 🏗️ Project Overview

This project demonstrates how to use **auto-decoder architecture** to learn a continuous latent space of 2D tower shapes. Unlike traditional autoencoders, auto-decoders learn latent codes directly as trainable parameters alongside the decoder network, enabling smooth interpolation and generation of new shape variations.

### Key Features

- 🧠 **Auto-Decoder Neural Architecture**: Learn latent codes and decoder jointly
- 🗼 **SDF Representation**: Tower shapes encoded as Signed Distance Fields
- 🎨 **Interactive Exploration**: PathSelect app for real-time latent space navigation
- 🎬 **Animation Generation**: Create smooth transitions between tower designs
- 📊 **Educational Tutorial**: Step-by-step learning implementation

## 📁 Project Structure

```
LantentSDF/
├── AutoDecoder.ipynb      # Main tutorial notebook
├── pathSelect.py          # Interactive exploration application
├── data.json             # Tower shape SDF data
├── README.md             # This file
├── models/               # Exported trained models (generated)
│   ├── decoder_model.h5
│   ├── latent_codes.npy
│   ├── coords_flat.npy
│   ├── latent_range.npy
│   └── data.json
└── output/               # Generated SDF sequences (generated)
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install tensorflow numpy matplotlib pygame
```

### 1. Run the Tutorial

Open and run `AutoDecoder.ipynb` to:
- Learn the auto-decoder architecture
- Train on tower shape data
- Explore the learned latent space
- Export the trained model

### 2. Interactive Exploration

After training, run the PathSelect application:

```bash
python pathSelect.py
```

**Controls:**
- **Left Click**: Add path points in latent space
- **Right Click**: Remove last path point
- **Mouse Hover**: Preview SDF at cursor position
- **SPACE**: Generate SDF sequence along path
- **S**: Save generated SDFs
- **C**: Clear all path points
- **H**: Show help screen

## 🎓 Tutorial Content

### Step 1: Auto-Decoder Architecture

Learn how auto-decoders work:
- **Latent Codes**: Trainable parameters representing each shape
- **Decoder Network**: Maps (latent_code + coordinate) → SDF_value
- **Joint Training**: Optimize both components simultaneously

### Step 2: SDF Representation

Understand Signed Distance Fields:
- **Negative values**: Inside the shape
- **Zero**: On the boundary
- **Positive values**: Outside the shape

### Step 3: Training Process

Train the model to learn latent representations:
- Load tower shape data from JSON
- Initialize learnable latent codes
- Train decoder network with reconstruction loss
- Visualize learning progress

### Step 4: Latent Space Exploration

Explore the learned continuous space:
- Visualize tower positions in 2D latent space
- Generate new tower variations by sampling
- Create smooth interpolations between designs

### Step 5: Interactive Application

Export and use the trained model:
- Save complete model package
- Run PathSelect for real-time exploration
- Create custom animation paths

## 🏛️ Architecture Details

### Auto-Decoder Network

```python
Input: [latent_code (2D) + coordinate (2D)] = 4D vector
↓
Dense(256, relu)
↓
Dense(256, relu)
↓
Dense(256, relu)
↓
Dense(256, relu)
↓
Dense(1, linear) → SDF value
```

### Training Objective

Minimize reconstruction loss across all shapes and coordinates:

```
Loss = MSE(predicted_SDF, target_SDF)
```

Jointly optimize:
- Latent codes: `L = {l₁, l₂, ..., lₙ}` 
- Decoder weights: `θ`

## 🎨 Key Concepts

### Latent Space Properties

- **Continuity**: Smooth transitions between nearby points
- **Interpolation**: Generate intermediate shapes
- **Generalization**: Sample novel tower variations
- **Compactness**: High-dimensional shapes → low-dimensional codes

### SDF Advantages

- **Implicit Representation**: Define shapes via distance functions
- **Resolution Independence**: Query at any coordinate
- **Smooth Boundaries**: Continuous gradients for optimization
- **Compact Storage**: Efficient shape encoding

## 📊 Results

The trained model achieves:
- **Faithful Reconstruction**: Original towers accurately reproduced
- **Smooth Interpolation**: Seamless transitions in latent space
- **Novel Generation**: Create new tower variations
- **Interactive Exploration**: Real-time latent space navigation

## 🔬 Technical Features

### Auto-Decoder Advantages

1. **No Encoder Required**: Direct latent code optimization
2. **Flexible Architecture**: Adapt to various shape types
3. **Continuous Representation**: Infinite resolution sampling
4. **Efficient Training**: Joint optimization strategy

### Implementation Highlights

- **TensorFlow/Keras**: Modern deep learning framework
- **Vectorized Operations**: Efficient batch processing
- **Interactive Visualization**: Real-time exploration tools
- **Export Pipeline**: Complete model packaging

## 🎯 Applications

### Educational

- **Neural Architecture Understanding**: Learn auto-decoder concepts
- **Latent Space Visualization**: 2D space for easy comprehension
- **Interactive Learning**: Hands-on exploration tools

### Research & Development

- **Shape Generation**: Create new design variations
- **Animation Creation**: Smooth morphing sequences
- **Design Space Exploration**: Navigate possibilities systematically
- **Architectural Studies**: Tower design optimization

### Creative Applications

- **Procedural Generation**: Automated tower creation
- **Game Assets**: Diverse building designs
- **Architectural Visualization**: Design space exploration
- **Art Generation**: Creative shape morphing

## 🛠️ Customization

### Modify Architecture

```python
# Change latent dimension
LATENT_DIM = 3  # 3D latent space

# Adjust network depth
layers.Dense(512, activation='relu')  # Wider layers
```

### Add New Shapes

1. Update `data.json` with new scalar fields
2. Retrain the model
3. Explore expanded latent space

### Export Formats

- **JSON**: SDF sequences for animation
- **NumPy**: Raw data for analysis
- **Images**: Visual sequences

## 📚 Learning Resources

### Concepts

- **DeepSDF Paper**: Original auto-decoder research
- **Implicit Neural Representations**: Modern shape encoding
- **Latent Space Methods**: Continuous representations
- **SDF Mathematics**: Distance field theory

### Extensions

- **3D Shapes**: Extend to volumetric SDFs
- **Style Transfer**: Combine shape and appearance
- **Conditional Generation**: Control specific attributes
- **Multi-Modal**: Combine with other representations

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- **3D Extension**: Volumetric tower shapes
- **Advanced Architectures**: Attention mechanisms, skip connections
- **New Applications**: Different shape types, materials
- **Visualization**: Enhanced interactive features

## 📝 License

This project is open source. Feel free to use, modify, and distribute for educational and research purposes.

## 🙏 Acknowledgments

- **DeepSDF**: Inspiration for auto-decoder architecture
- **TensorFlow Team**: Excellent deep learning framework
- **Community**: Open source tools and libraries

---

**Happy Learning and Exploring! 🎉**

*Discover the beauty of continuous shape spaces through interactive latent exploration.*
