# Dynamics Autoencoder Workspace

A comprehensive framework for simulating dynamical systems and learning their representations using autoencoders.

## Project Structure

```
autoencoder/
├── src/
│   ├── dynamics/          # Dynamics simulation modules
│   ├── models/            # Autoencoder architectures
│   ├── training/          # Training and evaluation pipelines
│   └── utils/             # Utility functions
├── configs/               # Configuration files
├── data/                  # Generated datasets
├── results/               # Training results and visualizations
├── notebooks/             # Jupyter notebooks for analysis
├── requirements.txt       # Python dependencies
└── main.py               # Main orchestration script
```

## Setup

### Automated Setup (Recommended)
```bash
# Run the setup script (handles both conda and venv)
chmod +x setup_env.sh
./setup_env.sh
```

### Manual Setup
1. Create and activate virtual environment:
```bash
# Using conda
conda create -n dynamics-ae python=3.9
conda activate dynamics-ae

# Or using venv
python -m venv dynamics-ae
source dynamics-ae/bin/activate  # On Windows: dynamics-ae\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Test the installation:
```bash
python test_basic.py
```

## Usage

### Quick Start
1. Run with the Lorenz system manifold learning:
```bash
python main.py --config configs/lorenz_config.yaml
```

2. Try other predefined systems:
```bash
# Rossler attractor
python main.py --config configs/rossler_config.yaml

# Van der Pol oscillator
python main.py --config configs/vanderpol_config.yaml
```

### Advanced Usage
- Skip data generation (use existing dataset):
```bash
python main.py --config configs/lorenz_config.yaml --skip-data-generation
```

- Specify device for training:
```bash
python main.py --config configs/lorenz_config.yaml --device cuda
```

## Workflow

1. **System Definition**: Define ODE dynamics in configuration
2. **Data Generation**: Simulate trajectories from initial condition boxes
3. **Dataset Creation**: Store trajectories as pickle files
4. **State-Space Autoencoder Training**: Learn manifold coordinates for individual state points
5. **Manifold Evaluation**: Analyze learned coordinate transformation and reconstruction quality

## Configuration

All system parameters are specified in YAML configuration files:
- Dynamics model parameters
- Initial condition specifications
- Trajectory generation settings
- State-space autoencoder architecture
- Training hyperparameters
- Manifold evaluation metrics 