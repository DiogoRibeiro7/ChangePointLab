# doc_generator.py
# MIT License
"""
Documentation generator for the Change-Point & State-Space Toolkit.
Generates cross-reference documentation and usage examples.
"""

import os
import re
import ast
import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


class ModuleDocGenerator:
    """
    Generates documentation with cross-references between modules.
    """
    
    def __init__(self, output_dir: str = "docs"):
        self.output_dir = Path(output_dir)
        self.modules = {}
        self.dependencies = {}
        self.algorithms = {}
        
    def scan_directory(self, directory: str) -> None:
        """
        Scan directory for Python modules.
        
        Parameters
        ----------
        directory : str
            Directory to scan
        """
        directory = Path(directory)
        for py_file in directory.glob("*.py"):
            if py_file.name.startswith("_") or py_file.name == "setup.py":
                continue
                
            module_name = py_file.stem
            self.modules[module_name] = py_file
            
    def parse_imports(self) -> None:
        """Parse imports to build dependency graph."""
        for module_name, file_path in self.modules.items():
            self.dependencies[module_name] = set()
            
            with open(file_path, 'r') as f:
                content = f.read()
                
            # Parse AST to find imports
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    # Regular imports
                    if isinstance(node, ast.Import):
                        for name in node.names:
                            if name.name in self.modules:
                                self.dependencies[module_name].add(name.name)
                    
                    # From imports
                    elif isinstance(node, ast.ImportFrom):
                        if node.module in self.modules:
                            self.dependencies[module_name].add(node.module)
            except SyntaxError:
                print(f"Error parsing {module_name}")
                
    def categorize_modules(self) -> None:
        """Categorize modules by functionality."""
        categories = {
            "core": ["bayesian_blocks", "edivisive", "kcp", "kcp_rff", "within_period_cpd"],
            "state_space": ["hsmm", "sdhmm", "sdhmm_mix_vi"],
            "emissions": ["gaussian_diag", "gaussian_full", "ar_emissions"],
            "advanced": ["rff_variants", "bandwidth_cv", "tempering"],
            "plotting": ["bb_plotting", "edivisive_plotting", "kcp_plotting", "plotting_helpers"],
            "utilities": ["utils", "types", "io_utils", "data_loader", "diagnostics"],
            "cli": ["cli", "cpd_cli"]
        }
        
        self.categories = {}
        for category, modules in categories.items():
            for module in modules:
                if module in self.modules:
                    self.categories[module] = category
    
    def extract_algorithm_info(self) -> None:
        """Extract algorithm information from modules."""
        for module_name, file_path in self.modules.items():
            with open(file_path, 'r') as f:
                content = f.read()
                
            # Extract algorithm description
            desc_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
            description = desc_match.group(1).strip() if desc_match else ""
            
            # Extract main functions
            func_matches = re.findall(r'def\s+([a-zA-Z0-9_]+)\s*\((.*?)\)', content)
            functions = [name for name, _ in func_matches if not name.startswith('_')]
            
            # Extract classes
            class_matches = re.findall(r'class\s+([a-zA-Z0-9_]+)(?:\s*\(.*?\))?:', content)
            classes = [name for name in class_matches if not name.startswith('_')]
            
            self.algorithms[module_name] = {
                "description": description,
                "functions": functions,
                "classes": classes,
                "category": self.categories.get(module_name, "other")
            }
    
    def generate_dependency_graph(self) -> None:
        """Generate dependency graph visualization."""
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
            
            G = nx.DiGraph()
            
            # Add nodes colored by category
            for module in self.modules:
                category = self.categories.get(module, "other")
                G.add_node(module, category=category)
            
            # Add edges
            for module, deps in self.dependencies.items():
                for dep in deps:
                    if dep in self.modules:
                        G.add_edge(module, dep)
            
            # Set up colors by category
            category_colors = {
                "core": "tab:red",
                "state_space": "tab:blue",
                "emissions": "tab:green",
                "advanced": "tab:purple",
                "plotting": "tab:orange",
                "utilities": "tab:gray",
                "cli": "tab:brown",
                "other": "tab:pink"
            }
            
            node_colors = [category_colors[self.categories.get(node, "other")] 
                          for node in G.nodes()]
            
            # Draw the graph
            plt.figure(figsize=(12, 10))
            pos = nx.spring_layout(G, seed=42)
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1000, alpha=0.8)
            nx.draw_networkx_edges(G, pos, edge_color="gray", width=1, arrowsize=15)
            nx.draw_networkx_labels(G, pos, font_size=8)
            
            # Create legend
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                         label=category.capitalize(),
                                         markerfacecolor=color, markersize=10)
                               for category, color in category_colors.items()]
            plt.legend(handles=legend_elements, loc='upper right')
            
            plt.title("Module Dependencies")
            plt.axis('off')
            
            # Save the graph
            self.output_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(self.output_dir / "dependency_graph.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except ImportError:
            print("NetworkX not available, skipping dependency graph")
    
    def generate_module_docs(self) -> None:
        """Generate documentation for each module."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate module index
        with open(self.output_dir / "index.md", 'w') as f:
            f.write("# Change-Point & State-Space Toolkit Documentation\n\n")
            f.write("This documentation provides cross-references between modules and detailed usage examples.\n\n")
            
            # Add dependency graph
            f.write("## Module Dependencies\n\n")
            f.write("![Module Dependencies](dependency_graph.png)\n\n")
            
            # Add module categories
            f.write("## Module Categories\n\n")
            categories_grouped = {}
            for module, category in self.categories.items():
                if category not in categories_grouped:
                    categories_grouped[category] = []
                categories_grouped[category].append(module)
            
            for category, modules in sorted(categories_grouped.items()):
                f.write(f"### {category.capitalize()}\n\n")
                for module in sorted(modules):
                    f.write(f"- [{module}]({module}.md): {self.algorithms[module]['description'].split('.')[0]}.\n")
                f.write("\n")
            
            # Add algorithm index
            f.write("## Algorithm Index\n\n")
            f.write("| Algorithm | Module | Category | Description |\n")
            f.write("|-----------|--------|----------|-------------|\n")
            
            for module, info in sorted(self.algorithms.items()):
                description = info['description'].split('.')[0]
                category = info['category'].capitalize()
                f.write(f"| [{module}]({module}.md) | `{module}` | {category} | {description} |\n")
            
    def generate_module_pages(self) -> None:
        """Generate individual module documentation pages."""
        for module_name, info in self.algorithms.items():
            with open(self.output_dir / f"{module_name}.md", 'w') as f:
                f.write(f"# {module_name}\n\n")
                
                # Module description
                f.write("## Description\n\n")
                f.write(f"{info['description']}\n\n")
                
                # Module category
                category = info['category'].capitalize()
                f.write(f"**Category**: {category}\n\n")
                
                # Dependencies
                deps = sorted(self.dependencies.get(module_name, []))
                if deps:
                    f.write("## Dependencies\n\n")
                    f.write("This module depends on:\n\n")
                    for dep in deps:
                        f.write(f"- [{dep}]({dep}.md)\n")
                    f.write("\n")
                
                # Dependent modules
                dependents = []
                for other_module, other_deps in self.dependencies.items():
                    if module_name in other_deps:
                        dependents.append(other_module)
                
                if dependents:
                    f.write("## Used By\n\n")
                    f.write("This module is used by:\n\n")
                    for dep in sorted(dependents):
                        f.write(f"- [{dep}]({dep}.md)\n")
                    f.write("\n")
                
                # Functions
                if info['functions']:
                    f.write("## Functions\n\n")
                    for func in sorted(info['functions']):
                        f.write(f"- `{func}`\n")
                    f.write("\n")
                
                # Classes
                if info['classes']:
                    f.write("## Classes\n\n")
                    for cls in sorted(info['classes']):
                        f.write(f"- `{cls}`\n")
                    f.write("\n")
                
                # Usage examples
                f.write("## Usage Examples\n\n")
                f.write("```python\n")
                f.write(self._generate_example(module_name, info))
                f.write("```\n\n")
                
                # Related modules
                related = [m for m in self.modules.keys() 
                          if self.categories.get(m) == info['category'] and m != module_name]
                
                if related:
                    f.write("## Related Modules\n\n")
                    for rel in sorted(related):
                        f.write(f"- [{rel}]({rel}.md): {self.algorithms[rel]['description'].split('.')[0]}.\n")
                    f.write("\n")
                
                # Return to index
                f.write("[Return to Index](index.md)\n")
    
    def _generate_example(self, module_name: str, info: dict) -> str:
        """Generate example code for a module."""
        examples = {
            # Core algorithms
            "bayesian_blocks": """
# For events data (unbinned Poisson)
from bayesian_blocks import bayesian_blocks_events
result = bayesian_blocks_events(t, t_start=0.0, t_stop=10.0, p0=0.05)
print(result.edges, result.block_value)

# For binned counts
from bayesian_blocks import bayesian_blocks_counts
result = bayesian_blocks_counts(counts, widths=None, p0=0.05)

# Plot the results
from bb_plotting import plot_blocks
plot_blocks(result.edges, result.block_value, x)
""",
            "edivisive": """
from edivisive import edivisive
import numpy as np

# Generate example data
X = np.random.randn(200, 3)
X[100:] += 2  # Add a shift at t=100

# Run E-Divisive
result = edivisive(X, 
                  alpha=1.0,           # Energy statistic parameter
                  min_size=30,         # Minimum segment size
                  R=499,               # Number of permutations
                  resample="circular-block-bootstrap",
                  block_size=None,     # Auto-determine
                  significance=0.05,   # Significance level
                  seed=123)

print(f"Detected change points: {result.change_points}")

# Plot the results
from edivisive_plotting import plot_edivisive_result
plot_edivisive_result(X, result, ['X1', 'X2', 'X3'])
""",
            "kcp": """
import numpy as np
from kcp import gram_rbf, build_kernel_prefix, kcp_penalized

# Generate example data
X = np.random.randn(200, 2)
X[50:100] += 2
X[150:] -= 2

# Create kernel matrix and prefix sums
K, gamma = gram_rbf(X)  # Automatic bandwidth selection with median heuristic
pref = build_kernel_prefix(K)

# Run penalized KCP
result = kcp_penalized(pref, 
                      gamma=np.log(X.shape[0]),  # BIC-like penalty
                      min_size=20,
                      method="pelt")  # Use PELT pruning for efficiency

print(f"Detected change points: {result.change_points}")

# Plot the results
from kcp_plotting import plot_kcp_result
plot_kcp_result(X, result, columns=['X1', 'X2'], kernel='rbf')
""",
            "kcp_rff": """
import numpy as np
from kcp_rff import RFFConfig, rbf_rff_map, build_feature_prefix, rff_kcp_penalized

# Generate example data
X = np.random.randn(500, 10)  # Higher dimension where RFF is beneficial
X[200:350] += 1.5

# Create RFF mapping
rff_config = RFFConfig(n_features=512, seed=123)
rff = rbf_rff_map(X, rff_config)  # Automatic bandwidth selection
pref = build_feature_prefix(rff.Z)

# Run penalized KCP on RFF features
result = rff_kcp_penalized(pref,
                          gamma_pen=np.log(X.shape[0]),
                          min_size=20,
                          method="pelt")

print(f"Detected change points: {result.change_points}")
""",
            "within_period_cpd": """
import numpy as np
from within_period_cpd import WithinPeriodCPD, ModelPrior, RJConfig
from data_loader import load_binary_from_csv

# Either load binary data from CSV timestamps
x, N = load_binary_from_csv("events.csv", 
                           timestamp_col="timestamp",
                           bin_minutes=15,
                           start_hour=0)

# Or create synthetic data
N = 96  # 15-minute bins over 24 hours
l = 4   # Minimum segment length (1 hour)
days = 30
x = np.random.binomial(1, p=0.2, size=N * days).astype(bool)

# Set up model and run MCMC
prior = ModelPrior(N=N, l=l, gamma=1.0, pois_lambda=1.0)
model = WithinPeriodCPD(prior)
cfg = RJConfig(iters=20000, burn=10000, thin=10, seed=123)

result = model.fit(x, cfg)
print(f"MAP changepoints: {result.mode_tau}")

# Compute posterior summaries
pw = model.pointwise_posterior_summary_from_samples(
    result.samples_tau, draws_per_sample=2, credible=0.95)

# Plot results
from plotting_helpers import plot_changepoint_posterior_mass, plot_pointwise_bands
plot_changepoint_posterior_mass(
    cp_hist=result.changepoint_hist,
    num_samples=len(result.samples_tau),
    N=N,
    tau_map=result.mode_tau,
    start_hour=0)
""",
            # State-space models
            "hsmm": """
import numpy as np
from hsmm import HSMM, HSMMConfig, HSMMParams, PoissonDur
from gaussian_diag import estimate_by_kmeanspp, gaussian_diag_loglik

# Generate or load data
X = np.random.randn(500, 3)  # Replace with your data

# Number of states
K = 3

# Initialize emissions using k-means++
em = estimate_by_kmeanspp(X, K, n_init=5, allow_nan=False)
L = gaussian_diag_loglik(X, em)

# Set up HSMM with Poisson durations
pi0 = np.full(K, 1.0/K)  # Initial state distribution
A0 = np.full((K,K), 1.0/(K-1))  # Transition matrix (excl. self-transitions)
np.fill_diagonal(A0, 0.0)  # No self-transitions in HSMM

# Configure HSMM
hsmm = HSMM(
    HSMMConfig(K=K, Dmax=150, min_duration=5),
    HSMMParams(
        pi=pi0, 
        A=A0, 
        duration=("poisson", PoissonDur(lam=np.array([60, 80, 70])))
    )
)

# Fit model
params_fit, ll_trace = hsmm.fit(L)

# Decode states
z_vit, d_vit = hsmm.decode_viterbi(L)
print(f"Decoded states: {np.unique(z_vit)}")
print(f"Average durations: {[d_vit[z_vit == k].mean() for k in range(K)]}")
""",
            "gaussian_diag": """
import numpy as np
from gaussian_diag import (
    GaussianDiagParams, gaussian_diag_loglik, 
    estimate_from_labels, estimate_from_responsibilities,
    estimate_by_kmeanspp
)

# Generate or load data
X = np.random.randn(500, 3)  # Replace with your data
K = 3  # Number of components/states

# Initialize with k-means++
params = estimate_by_kmeanspp(X, K, n_init=5, allow_nan=False)
print(f"Means: {params.means}")
print(f"Variances: {params.vars}")

# Compute log-likelihoods
loglik = gaussian_diag_loglik(X, params)  # Shape: (T, K)

# Convert to responsibilities (for EM)
max_ll = np.max(loglik, axis=1, keepdims=True)
resp = np.exp(loglik - max_ll)
resp = resp / np.sum(resp, axis=1, keepdims=True)

# Update parameters from responsibilities
updated_params = estimate_from_responsibilities(X, resp)
""",
            "gaussian_full": """
import numpy as np
from gaussian_full import (
    GaussianFullParams, gaussian_full_loglik,
    estimate_gaussian_full_from_labels,
    estimate_gaussian_full_from_responsibilities,
    estimate_gaussian_full_by_kmeans,
    GaussianFullEmissions
)

# Generate or load data
X = np.random.randn(500, 3)  # Replace with your data
K = 3  # Number of components/states

# Initialize with k-means
params = estimate_gaussian_full_by_kmeans(X, K, n_init=5, seed=42)
print(f"Means: {params.means}")
print(f"Covariances shape: {params.covs.shape}")

# Compute log-likelihoods
loglik = gaussian_full_loglik(X, params)  # Shape: (T, K)

# Alternative: use the emissions class
emissions = GaussianFullEmissions(n_states=K)
emissions.initialize_kmeans(X, n_init=5)
loglik = emissions.compute_loglik(X)

# Update from responsibilities (for EM)
responsibilities = np.ones((X.shape[0], K)) / K  # Replace with actual responsibilities
emissions.update_from_responsibilities(X, responsibilities)
""",
            "ar_emissions": """
import numpy as np
from ar_emissions import (
    ARParams, ar_loglik, estimate_ar_from_labels,
    estimate_ar_from_responsibilities, simulate_ar_process,
    AREmissions
)

# Generate or load data
T, D, K = 500, 2, 3  # Time points, dimensions, states
X = np.random.randn(T, D)  # Replace with your data

# True or estimated state sequence
states = np.random.choice(K, size=T)  # Replace with actual states

# Estimate AR parameters from labels
ar_order = 2  # AR(2) process
params = estimate_ar_from_labels(X, states, K, order=ar_order)

# Compute log-likelihoods
loglik = ar_loglik(X, params)  # Shape: (T, K), first 'order' points get -inf

# Simulate from the model
simulated_X = simulate_ar_process(params, states, seed=42)

# Alternative: use the emissions class
emissions = AREmissions(n_states=K, order=ar_order)
emissions.initialize(X, method="kmeans", seed=42)
loglik = emissions.compute_loglik(X)

# Update from responsibilities (for EM)
responsibilities = np.ones((X.shape[0], K)) / K  # Replace with actual responsibilities
emissions.update_from_responsibilities(X, responsibilities)
""",
            # Advanced
            "rff_variants": """
import numpy as np
from rff_variants import (
    OrthogonalRFFConfig, QuasiMCRFFConfig, CompactRFFConfig,
    orthogonal_rff_map, quasi_mc_rff_map, compact_support_rff_map,
    compare_rff_variants, adaptive_rff_map
)

# Generate or load data
X = np.random.randn(500, 10)  # Replace with your data

# Orthogonal RFF (reduced variance)
orth_config = OrthogonalRFFConfig(n_features=512, structured=True, seed=42)
orth_rff = orthogonal_rff_map(X, orth_config)
print(f"Orthogonal RFF features shape: {orth_rff.Z.shape}")

# Quasi-Monte Carlo RFF (better coverage)
qmc_config = QuasiMCRFFConfig(n_features=512, sequence_type="sobol", seed=42)
qmc_rff = quasi_mc_rff_map(X, qmc_config)

# Compact support RFF (local structure)
compact_config = CompactRFFConfig(n_features=512, support_radius=2.0, seed=42)
compact_rff = compact_support_rff_map(X, compact_config)

# Adaptive RFF (automatic feature count)
adaptive_rff = adaptive_rff_map(X, base_features=128, max_features=1024, tolerance=1e-3)
print(f"Adaptive RFF selected {adaptive_rff.config['final_features']} features")

# Compare variants
comparison = compare_rff_variants(X, n_features=512, seed=42)
for variant, metrics in comparison.items():
    print(f"{variant}: MSE={metrics['mse']:.6f}")
""",
            "bandwidth_cv": """
import numpy as np
from bandwidth_cv import (
    select_rbf_bandwidth_cv, select_rbf_bandwidth_information_criterion,
    select_rbf_bandwidth_multiscale, bandwidth_stability_analysis,
    BandwidthCVConfig
)

# Generate or load data
X = np.random.randn(200, 3)  # Replace with your data

# K-fold cross-validation for bandwidth
config = BandwidthCVConfig(
    method="kfold",
    cv_folds=5,
    search_strategy="grid",
    n_candidates=15,
    sigma_range=(0.1, 10.0),
    log_space=True,
    scoring="likelihood",
    seed=42
)
sigma_cv = select_rbf_bandwidth_cv(X, config)
print(f"CV-selected bandwidth: σ = {sigma_cv:.4f}")

# Information criterion (AIC/BIC)
sigma_bic = select_rbf_bandwidth_information_criterion(X, criterion="bic")
print(f"BIC-selected bandwidth: σ = {sigma_bic:.4f}")

# Multi-scale bandwidth selection
multiscale = select_rbf_bandwidth_multiscale(X, n_scales=3)
for scale, sigma in multiscale.items():
    print(f"{scale}: σ = {sigma:.4f}")

# Stability analysis
stability = bandwidth_stability_analysis(X, n_bootstrap=20)
print(f"Bandwidth stability: {stability['coefficient_of_variation']:.3f}")
""",
            "tempering": """
import numpy as np
from tempering import PTConfig, parallel_tempering_fit
from within_period_cpd import WithinPeriodCPD, ModelPrior

# Set up model (using within-period CPD as an example)
N = 96  # 15-minute bins over 24 hours
l = 4   # Minimum segment length (1 hour)
prior = ModelPrior(N=N, l=l, gamma=1.0, pois_lambda=1.0)
model = WithinPeriodCPD(prior)

# Generate or load binary data
x = np.random.binomial(1, p=0.2, size=N * 30).astype(bool)

# Configure parallel tempering
ptcfg = PTConfig(
    iters=20000,
    burn=10000,
    thin=10,
    swap_every=50,
    T_hot=3.0,  # Temperature of hot chain
    seed=42
)

# Run parallel tempering
ptres = parallel_tempering_fit(model, x, ptcfg)

print(f"MAP changepoints: {ptres.mode_tau_cold}")
print(f"Swap acceptance rate: {ptres.swaps_accepted / ptres.swaps_attempted:.2f}")

# Use cold chain samples for analysis
samples = ptres.samples_tau_cold
log_posts = ptres.log_posts_cold
""",
            # Utilities
            "utils": """
import numpy as np
from utils import (
    log_beta, safe_normalize, softmax, logsumexp,
    ensure_psd, stable_logdet_inv, mahalanobis_distance,
    build_prefix_sum_1d, range_sum_1d,
    sliding_window, autocorr,
    kmeanspp_init, median_heuristic
)

# Statistical utilities
beta_val = log_beta(2.0, 3.0)
probs = softmax(np.array([1.0, 2.0, 0.5]))

# Matrix operations
cov = np.array([[1.0, 0.5], [0.5, 0.8]])
psd_cov = ensure_psd(cov)
logdet, inv_cov = stable_logdet_inv(psd_cov)

# Distance computation
x = np.array([1.0, 2.0])
mean = np.array([0.0, 0.0])
dist = mahalanobis_distance(x, mean, inv_cov)

# Prefix sums for O(1) range queries
arr = np.arange(10)
prefix = build_prefix_sum_1d(arr)
range_sum = range_sum_1d(prefix, 2, 7)  # Sum arr[2:7]

# Time series utilities
X = np.random.randn(100)
windows = sliding_window(X, window_size=10, step=5)
autocorrelation = autocorr(X, max_lag=20)

# Sampling and initialization
X_data = np.random.randn(200, 3)
centers = kmeanspp_init(X_data, k=5, random_state=42)
bandwidth = median_heuristic(X_data)
""",
            "types": """
from types import (
    Array1D, Array1DFloat, ArrayBool, Tau,
    RJConfig, PTConfig, MCMCResult, PTResult, ChangePointResult
)

# Use standard type aliases
points = Array1DFloat(shape=(100,))
segments = ArrayBool(shape=(100,))
changepoints = Tau((25, 50, 75))  # Tuple of integer changepoints

# Use configuration classes
rj_config = RJConfig(
    iters=20000,
    burn=10000,
    thin=10,
    seed=42,
    move_prob=0.5,
    birth_prob=0.25,
    death_prob=0.25
)

pt_config = PTConfig(
    iters=20000,
    burn=10000,
    thin=10,
    swap_every=50,
    T_hot=3.0,
    seed=42
)

# Standardized result container
result = ChangePointResult(
    change_points=[25, 50, 75],
    segments=[(0, 25), (25, 50), (50, 75), (75, 100)],
    scores=[0.95, 0.92, 0.88],
    cost=10.5,
    model_name="kcp",
    parameters={"gamma": 1.0}
)
""",
            "io_utils": """
import numpy as np
from io_utils import save_result_npz, load_result_npz

# Assuming we have MCMC results from within_period_cpd
from within_period_cpd import ModelPrior, RJConfig, WithinPeriodCPD

# Example result components
samples_tau = [(25, 75), (26, 74), (25, 76)]  # List of Tau tuples
log_posteriors = [-105.2, -106.3, -104.8]
changepoint_hist = np.zeros(96, dtype=np.int64)
for tau in samples_tau:
    for cp in tau:
        changepoint_hist[cp] += 1
mode_tau = (25, 75)  # The highest posterior sample

# Prior and config used
prior = ModelPrior(N=96, l=4, gamma=1.0, pois_lambda=1.0)
cfg = RJConfig(iters=20000, burn=10000, thin=10, seed=42)

# Save results to NPZ
save_result_npz(
    "results.npz",
    samples_tau=samples_tau,
    log_posteriors=log_posteriors,
    changepoint_hist=changepoint_hist,
    mode_tau=mode_tau,
    prior_obj=prior,
    cfg_obj=cfg
)

# Load results
loaded = load_result_npz("results.npz")
print(f"Loaded {len(loaded['samples_tau'])} samples")
print(f"MAP changepoints: {loaded['mode_tau']}")
""",
            "data_loader": """
import numpy as np
from data_loader import load_binary_from_csv, empirical_per_bin_mean

# Load binary events from timestamps in a CSV file
x, N = load_binary_from_csv(
    "events.csv",
    timestamp_col="timestamp",  # Column with ISO timestamps
    value_col=None,             # Optional value column to threshold
    value_threshold=0.0,        # Threshold for value_col
    bin_minutes=15,             # Time resolution (15-min bins)
    start_hour=0,               # Hour corresponding to bin 0
    days_span=None              # Optional exact number of days
)

# Compute empirical per-bin probabilities (averaged across days)
empirical_probs = empirical_per_bin_mean(x, N)

print(f"Loaded {len(x)} binary observations")
print(f"Period length N = {N} bins")
print(f"Average activity rates: min={empirical_probs.min():.3f}, max={empirical_probs.max():.3f}")
""",
            "diagnostics": """
import numpy as np
from diagnostics import (
    PosteriorM, posterior_num_segments,
    autocorr_1d, ess_geyer, ess_for_cp_indicator
)

# Assuming we have MCMC samples from within_period_cpd
samples_tau = [(25, 75), (26, 74), (25, 76), (26, 75), (25, 75)]
N = 96  # Period length

# Analyze posterior over number of segments m
pm = posterior_num_segments(samples_tau)
print(f"Posterior over m: {list(zip(pm.m_values, pm.probs))}")

# Compute autocorrelation of a chain
chain = np.random.randn(1000)  # Replace with actual MCMC chain values
acf = autocorr_1d(chain, max_lag=50)

# Compute effective sample size
ess = ess_geyer(chain)
print(f"Effective sample size: {ess:.1f} out of {len(chain)}")

# Compute ESS for changepoint indicators
cp_ess = ess_for_cp_indicator(samples_tau, N)
print(f"Min ESS across changepoint positions: {np.min(cp_ess):.1f}")
print(f"Median ESS across changepoint positions: {np.median(cp_ess):.1f}")
""",
            # CLI tools
            "cli": """
# Run from command line:

# Synthetic demo (RJMCMC)
python -m cli --demo --N 96 --l 4 --days 30 --iters 20000 --burn 10000 --thin 10 --outdir out

# Synthetic demo using Parallel Tempering
python -m cli --demo --N 96 --l 4 --days 30 --pt --T-hot 3.0 --swap-every 50 --iters 20000 --burn 10000 --thin 10 --outdir out_pt

# Fit from CSV (timestamp column "ts", 15-min bins, day starts at 00:00)
python -m cli --csv events.csv --timestamp-col ts --bin-minutes 15 --start-hour 0 --l 4 --iters 20000 --burn 10000 --thin 10 --outdir out_csv

# Load a previous run and just plot
python -m cli --load out/run.npz --plot-only --outdir out_plots
""",
            "cpd_cli": """
# Run from command line:

# Bayesian Blocks for events
python cpd_cli.py bayesian-blocks --input events.csv --timestamp-col time --output results/

# E-Divisive multivariate CPD
python cpd_cli.py edivisive --input data.csv --columns x,y,z --output results/

# Kernel CPD with RBF
python cpd_cli.py kcp --input data.csv --kernel rbf --output results/

# RFF KCP for large datasets
python cpd_cli.py rff-kcp --input data.csv --n-features 512 --output results/

# HSMM with Gaussian emissions
python cpd_cli.py hsmm --input data.csv --n-states 3 --emission gaussian --output results/

# Within-period CPD for daily patterns
python cpd_cli.py within-period --input activity.csv --bin-minutes 15 --output results/
""",
            # Default example for other modules
            "default": """
# Example code for this module:
import numpy as np
from {module_name} import *

# Please refer to the module description and functions list
# for details about the available functionality.
"""
        }
        
        # Return the example for the module or default if not defined
        if module_name in examples:
            return examples[module_name].strip()
        else:
            return examples["default"].replace("{module_name}", module_name).strip()
    
    def run(self) -> None:
        """Run the documentation generator pipeline."""
        print("Scanning modules...")
        self.scan_directory(".")
        
        print(f"Found {len(self.modules)} modules")
        
        print("Parsing imports...")
        self.parse_imports()
        
        print("Categorizing modules...")
        self.categorize_modules()
        
        print("Extracting algorithm information...")
        self.extract_algorithm_info()
        
        print("Generating dependency graph...")
        self.generate_dependency_graph()
        
        print("Generating module docs...")
        self.generate_module_docs()
        
        print("Generating module pages...")
        self.generate_module_pages()
        
        print(f"Documentation generated in {self.output_dir}")


if __name__ == "__main__":
    generator = ModuleDocGenerator(output_dir="docs")
    generator.run()
