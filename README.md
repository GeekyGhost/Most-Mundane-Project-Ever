# ComfyUI Quantum Video Trainer

**First-principles multimodal video training with Variational Free Energy optimization**

A ComfyUI custom node suite that enables incremental multimodal video training using first principles (music theory, physics, linguistics) combined with Variational Free Energy optimization and evolutionary algorithms.

## 🎯 Key Features

- **Incremental Learning**: Train on 10 videos at a time, accumulating knowledge without ever resetting
- **Theory-Guided**: Uses music theory (Circle of Fifths), physics (motion models), and editing theory
- **VFE Optimization**: Variational Free Energy and Expected Free Energy for intelligent learning
- **Evolutionary Weights**: Genetic algorithms optimize theory weight combinations
- **Persistent State**: Never lose training progress - knowledge base saved between sessions
- **ComfyUI Native**: Fully integrated node-based visual programming workflow

## 📦 Installation

### Prerequisites

- ComfyUI installed and working
- Python 3.8+
- PyTorch 2.0+

### Install Steps

```bash
# Navigate to ComfyUI custom_nodes directory
cd ComfyUI/custom_nodes/

# Clone this repository
git clone https://github.com/GeekyGhost/ComfyUI-QuantumVideoTrainer.git

# Install dependencies
cd ComfyUI-QuantumVideoTrainer
pip install -r requirements.txt

# Restart ComfyUI
```

## 🚀 Quick Start

### 1. Prepare Your Videos

Create a directory with 5-second video clips:

```bash
mkdir test_videos
# Add 10 diverse video clips (*.mp4)
```

### 2. Basic Training Workflow

In ComfyUI, create this node graph:

```
QVT_LoadVideoBatch ──> QVT_LoadKnowledgeBase ──> QVT_IncrementalTrain ──> QVT_SaveKnowledgeBase
     │                        │                           │
     │                        │                           └──> Training Report (text output)
     └────────────────────────┘
```

**Node Configuration:**
- **QVT_LoadVideoBatch**:
  - `video_directory`: "./test_videos"
  - `pattern`: "*.mp4"
  - `max_videos`: 10
  - `max_duration`: 5.0

- **QVT_LoadKnowledgeBase**:
  - `knowledge_base_path`: "./knowledge_base.pkl"
  - `create_if_missing`: True

- **QVT_IncrementalTrain**:
  - `generations`: 50
  - `beta`: 0.7 (balance VFE/EFE)

- **QVT_SaveKnowledgeBase**:
  - `save_path`: "./knowledge_base.pkl"

### 3. Run Training

Execute the workflow. You'll see:
- Video decomposition progress
- Theory mapping analysis
- Evolution fitness scores
- Updated knowledge base

## 🧠 How It Works

### Theory-Guided Learning

The system maps videos to multiple theory spaces:

#### 🎵 Music Theory
- **Circle of Fifths**: Analyzes harmonic progressions
- **Jazz Extensions**: Handles complex chord movements
- Scores: 0.0-1.0 adherence to music theory

#### ⚛️ Physics Models
- **Smooth Motion**: Temporal continuity
- **Lorenz Attractor**: Chaotic but structured motion
- **Orbital**: Circular/elliptical patterns
- **Chaos**: Pure unpredictability

#### 🎬 Editing Theory
- **Continuity Editing**: Traditional cuts
- **Montage Editing**: Rhythmic juxtaposition
- Analyzes cut patterns and transitions

### Variational Free Energy (VFE)

The system minimizes free energy:

```
VFE = KL(q || p) + E_q[-log p(y|h)]
     ↑              ↑
     |              └─ Negative log-likelihood (prediction error)
     └─ KL divergence (complexity penalty)
```

- **Low VFE** = Good match to learned patterns
- **High VFE** = Surprising/unexpected content

### Evolutionary Optimization

Theory weights evolve through genetic algorithm:

1. **Population**: Start with current best weights
2. **Fitness**: Evaluate using VFE on video batch
3. **Selection**: Keep top 30% performers
4. **Crossover**: Combine parent weights
5. **Mutation**: Random perturbations
6. **Iteration**: Repeat for N generations

### Incremental Update

After each batch:
- **Bayesian Prior Update**: Gaussian and categorical priors refined
- **Knowledge Accumulation**: Video fingerprints added to history
- **State Persistence**: Everything saved to knowledge base

## 📊 Node Reference

### Input Nodes

#### 📹 QVT_LoadVideo
Load a single video file.

**Inputs:**
- `video_path`: Path to video file
- `max_duration`: Duration to load (seconds)
- `start_time`: Start position (seconds)

**Outputs:**
- `video`: Video tensor (T, H, W, C)
- `metadata`: Video metadata dict

#### 📹 QVT_LoadVideoBatch
Load multiple videos for batch training.

**Inputs:**
- `video_directory`: Directory path
- `pattern`: Glob pattern (e.g., "*.mp4")
- `max_videos`: Maximum videos to load
- `max_duration`: Duration per video

**Outputs:**
- `video_batch`: List of video tensors
- `batch_metadata`: List of metadata dicts

#### 💾 QVT_LoadKnowledgeBase
Load or create knowledge base.

**Inputs:**
- `knowledge_base_path`: Path to .pkl file
- `create_if_missing`: Create new if not found

**Outputs:**
- `knowledge_base`: Knowledge base object
- `summary`: Text summary

### Decomposition Nodes

#### 🔍 QVT_DecomposeVideo
Decompose video into multimodal features.

**Inputs:**
- `video`: Video tensor
- `metadata`: Video metadata
- `sample_fps`: Frames per second to sample
- `extract_audio`: Enable audio extraction
- `extract_music`: Enable music analysis
- `extract_motion`: Enable motion analysis

**Outputs:**
- `decomposed`: Decomposed video object
- `summary`: Text summary

### Theory Nodes

#### 🎵 QVT_MapToCircleOfFifths
Map music to circle of fifths theory.

**Inputs:**
- `decomposed`: Decomposed video
- `strict_mode`: Strict adherence vs jazz allowances

**Outputs:**
- `harmonic_profile`: Analysis dict
- `adherence_score`: 0.0-1.0 score

#### ⚛️ QVT_MapToPhysics
Map motion to physics models.

**Inputs:**
- `decomposed`: Decomposed video
- `physics_model`: ["smooth_motion", "lorenz_attractor", "orbital", "chaos"]

**Outputs:**
- `physics_profile`: Analysis dict
- `conformity_score`: 0.0-1.0 score

### VFE Nodes

#### 📊 QVT_ComputeVFE_Gaussian
Compute VFE for continuous features.

**Inputs:**
- `mu_q`: Posterior mean
- `Sigma_q`: Posterior covariance
- `mu_p`: Prior mean
- `Sigma_p`: Prior covariance
- `observation` (optional): Observed data
- `observation_noise` (optional): Observation covariance

**Outputs:**
- `vfe`: Total VFE value
- `kl_term`: KL divergence component
- `nll_term`: Negative log-likelihood

#### 📊 QVT_ComputeVFE_Categorical
Compute VFE for discrete features.

**Inputs:**
- `q_h`: Posterior distribution
- `p_h`: Prior distribution
- `p_y_given_h` (optional): Likelihood matrix
- `observation` (optional): Observed state index

**Outputs:**
- `vfe`: Total VFE value
- `kl_term`: KL divergence
- `nll_term`: Negative log-likelihood

### Training Nodes

#### 🎓 QVT_IncrementalTrain
**The core training orchestrator** - ties everything together.

**Inputs:**
- `knowledge_base`: Current knowledge base
- `video_batch`: Batch of videos
- `batch_metadata`: Metadata list
- `generations`: Evolution generations (default: 50)
- `beta`: VFE/EFE balance (0.0-1.0, default: 0.7)

**Outputs:**
- `updated_kb`: Updated knowledge base
- `training_report`: Detailed training summary
- `fitness_history`: Array of fitness scores

**Beta Parameter:**
- `beta=1.0`: Pure exploitation (match known patterns)
- `beta=0.0`: Pure exploration (seek novel information)
- `beta=0.7`: Balanced (recommended)

### State Nodes

#### 💾 QVT_SaveKnowledgeBase
Save knowledge base to disk.

**Inputs:**
- `knowledge_base`: Knowledge base object
- `save_path`: Path to save .pkl file

**Outputs:**
- `status`: Save confirmation text

## 🔬 Advanced Usage

### Custom Theory Weights

To manually set theory weights before training:

```python
# In Python, before loading into ComfyUI
from core.knowledge_base import KnowledgeBase

kb = KnowledgeBase('./my_kb.pkl')
kb.initialize()

# Emphasize jazz over classical harmony
kb.state['theory_weights']['jazz_extensions_weight'] = 0.8
kb.state['theory_weights']['circle_of_fifths_weight'] = 0.2

kb.save()
```

### Analyzing Results

```python
# Load saved knowledge base
kb = KnowledgeBase('./knowledge_base.pkl')
kb.load()

# View evolution history
import matplotlib.pyplot as plt
import numpy as np

fitness = [r['fitness'] for r in kb.state['evolution_history']]
plt.plot(fitness)
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('Evolution Progress')
plt.show()

# View learned priors
print(kb.state['vfe_priors_gaussian'])
print(kb.state['vfe_priors_categorical'])
```

## 🎓 Theory Background

### Why Variational Free Energy?

VFE comes from neuroscience's **Free Energy Principle** (Karl Friston):

> "The brain minimizes surprise by building generative models of the world"

In this system:
- **Generative Model**: Theory-based expectations (music, physics, editing)
- **Inference**: Video decomposition and theory mapping
- **Learning**: Update priors to reduce future surprise

### Why Evolutionary Algorithms?

Theory weights form a complex fitness landscape:
- **Non-differentiable**: Can't use gradient descent
- **Multi-modal**: Multiple valid solutions
- **Robust**: Genetic algorithms handle noisy fitness

### Why Incremental Learning?

Traditional deep learning requires:
- Thousands of examples
- GPU clusters
- Starting from scratch each time

This system enables:
- Learn from 10 videos at a time
- CPU-friendly (no backprop)
- Accumulate knowledge forever

## 🗺️ Roadmap

### v0.2.0 (Next Release)
- [ ] Video generation nodes
- [ ] Visualization nodes (theory space plots)
- [ ] Audio extraction using librosa
- [ ] Example workflow JSON files

### v0.3.0
- [ ] Real-time video analysis
- [ ] Custom theory module API
- [ ] Multi-GPU support for batch processing
- [ ] Web dashboard for knowledge base inspection

### v1.0.0
- [ ] Full audio-visual synchronization analysis
- [ ] Linguistic theory integration (if subtitles)
- [ ] Automated theory discovery
- [ ] Production-ready generation pipeline

## 🤝 Contributing

Contributions welcome! Areas of interest:

- **New theory modules**: Color theory, narrative structure, etc.
- **Better music analysis**: Actual librosa integration
- **Generation capabilities**: Use learned weights to create videos
- **Visualization**: Plot theory spaces, evolution, etc.

## 📄 License

MIT License - see LICENSE file

## 🙏 Acknowledgments

- **Karl Friston**: Free Energy Principle
- **ComfyUI Team**: Amazing node-based framework
- **Active Inference Community**: VFE/EFE formulations
- **Music Theory Community**: Circle of Fifths pedagogy

## 📚 References

1. Friston, K. (2010). The free-energy principle: a unified brain theory?
2. Active Inference: https://www.activeinference.org/
3. ComfyUI: https://github.com/comfyanonymous/ComfyUI
4. Circle of Fifths: https://en.wikipedia.org/wiki/Circle_of_fifths

## 💬 Support

- **Issues**: https://github.com/GeekyGhost/ComfyUI-QuantumVideoTrainer/issues
- **Discussions**: https://github.com/GeekyGhost/ComfyUI-QuantumVideoTrainer/discussions

---

**Built with ❤️ for the ComfyUI and Active Inference communities**
