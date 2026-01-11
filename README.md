# Quantum Amplitude Estimation for Space Debris Cascade Prediction

## Overview

This project explores the application of **Quantum Amplitude Estimation (QAE)** to space debris collision prediction, specifically focusing on the **Kessler Syndrome** - the cascading collision phenomenon in Low Earth Orbit (LEO).

## Key Insight: Path-Dependent Problems

Through extensive discussion and analysis, we discovered that QAE has a specific "sweet spot":

### When QAE Doesn't Help
- **Simple probability estimation**: Sobol/QMC sequences achieve O(1/N) convergence
- **Single-step Monte Carlo**: Classical methods are already efficient
- **State preparation overhead** negates quantum speedup for simple problems

### When QAE Excels
- **Path-dependent problems** with exponential path explosion
- **Conditional probability chains**: P(A|B) × P(B|C) × P(C|D) × ...
- **Multi-step simulations** where each step affects the next

```
Classical: 2^n paths to enumerate
Quantum:   √(2^n) = 2^(n/2) queries
```

This is analogous to **barrier options** in finance - where the option value depends on the entire price path, not just the final price.

## The Kessler Connection

The Kessler Syndrome is fundamentally a path-dependent problem:

1. **Each collision creates debris** → increases density
2. **Higher density → higher collision probability** (feedback loop)
3. **Cascade paths grow exponentially** with time steps
4. **Reaching critical threshold** = catastrophic runaway

This maps perfectly to the barrier option structure:
- Time steps = trading days
- Collision events = price movements
- Critical debris threshold = barrier level
- Cascade probability = knock-in option value

## Collision Avoidance Reality Check

Modern satellites (like Starlink) have **automated collision avoidance** systems:

| Collision Type | Avoidance Possible | Risk Level |
|---------------|-------------------|------------|
| Active-Active | Both can maneuver | Very Low |
| Active-Debris | Active maneuvers | Low |
| Active-Failed | Active maneuvers | Low |
| Debris-Debris | Neither can | **HIGH** |
| Failed-Failed | Neither can | **HIGH** |

**Key insight**: Kessler Syndrome doesn't start from satellite-satellite collisions. It starts when **debris-to-debris** collisions dominate - because neither object can avoid.

The model includes:
- Avoidance success rate (default 95% for Starlink-class)
- Failed satellite ratio (5%)
- Collision probability by type (avoidable vs unavoidable)

## Project Structure

```
qae/
├── Core Modules
│   ├── tle_fetcher.py          # CelesTrak TLE data fetcher
│   ├── orbit_propagator.py     # SGP4 orbit propagation
│   ├── collision_detector.py   # KD-Tree based collision detection
│   └── collision_probability.py # NASA-style Pc calculation
│
├── Quantum Modules
│   ├── quantum_collision.py    # Basic QAE for collision probability
│   ├── qae_path_option.py      # Path-dependent option pricing (demo)
│   ├── kessler_cascade_qae.py  # Kessler Syndrome QAE simulation
│   └── quantum_cluster_collision.py # Grover search for cluster pairs
│
├── Analysis
│   ├── cascade_cluster.py      # Spatial clustering for cascade risk
│   ├── analyze_debris.py       # Debris field analysis
│   └── visualizer.py           # Matplotlib visualizations
│
└── main.py                     # Main entry point
```

## Key Files Explained

### `kessler_cascade_qae.py`
The main Kessler Syndrome simulator using QAE approach:
- Models cascade as path-dependent conditional probability
- Includes collision avoidance system effects
- Compares classical enumeration vs quantum amplitude estimation
- Analyzes "tipping point" where avoidance becomes ineffective

### `qae_path_option.py`
Demonstrates the path-dependent QAE concept using barrier options:
- Binomial model for price paths
- Barrier knock-in/knock-out conditions
- Shows exponential path explosion problem
- Validates QAE speedup for path-dependent problems

### `cascade_cluster.py`
Classical spatial clustering to identify high-risk regions:
- DBSCAN-based clustering
- Multi-scale analysis (50km, 200km, 500km)
- Risk scoring based on debris ratio and density

## Installation

```bash
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- qiskit >= 1.0
- qiskit-aer
- sgp4
- numpy
- scipy
- requests
- matplotlib

## Usage

### Run Kessler Cascade Analysis
```bash
python kessler_cascade_qae.py
```

This runs:
1. Basic cascade probability estimation
2. Collision avoidance effectiveness comparison
3. Tipping point analysis

### Run Path-Dependent Option Demo
```bash
python qae_path_option.py
```

### Run Classical Cluster Detection
```bash
python cascade_cluster.py
```

## Results Summary

### QAE Speedup for Path-Dependent Problems

| Time Steps | Classical Ops | Quantum Queries | Speedup |
|------------|--------------|-----------------|---------|
| 5 | 32 | 5-6 | ~6x |
| 10 | 1,024 | 32 | ~32x |
| 15 | 32,768 | 181 | ~181x |
| 20 | 1,048,576 | 1,024 | ~1,024x |

### Collision Avoidance Effectiveness

With 95% avoidance success rate (Starlink-class):
- Active satellite collisions: Reduced by >90%
- But debris-debris collisions: **Unchanged**
- Tipping point: When debris-debris exceeds ~70% of total collision risk

## Conclusions

1. **QAE is not universally better** - it has specific use cases
2. **Path-dependent problems** are QAE's sweet spot
3. **Kessler Syndrome** maps perfectly to this structure
4. **Collision avoidance** changes the game but has limits
5. **The real danger** is debris-to-debris collision dominance

## Future Work

- Implement full IQAE (Iterative QAE) for better accuracy
- Add orbital mechanics to path probabilities
- Integrate with real Space Surveillance Network data
- Extend to multiple orbital shells
- Add decay and deorbit modeling

## References

- Kessler, D.J. & Cour-Palais, B.G. (1978). "Collision Frequency of Artificial Satellites"
- Brassard, G., et al. (2002). "Quantum Amplitude Amplification and Estimation"
- ESA Space Debris Office Reports
- CelesTrak (celestrak.org) for TLE data

## License

MIT License

## Acknowledgments

This project was developed through an exploratory conversation about finding practical applications for Quantum Amplitude Estimation, leading to the insight that path-dependent problems with exponential branching are the ideal use case.
