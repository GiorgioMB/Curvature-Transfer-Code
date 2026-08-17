import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Computer Modern Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.figsize': (5.5, 3.5),
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.5,
    'lines.markersize': 5
})

df = pd.read_csv('data/benchmark_runtime_scaling.csv')

fig, ax = plt.subplots()

ax.plot(df['n'], df['ot_time_sec'], marker='o', linestyle='-', color='#0072B2', label='Exact Ollivier-Ricci')
ax.plot(df['n'], df['bounds_time_sec'], marker='s', linestyle='--', color='#D55E00', label='Proposed Bounds')

ax.set_xlabel('Number of Nodes ($n$)')
ax.set_ylabel('Execution Time (s)')

ax.set_xscale('log')
ax.set_yscale('log')

ax.grid(True, which="major", linestyle='-', alpha=0.5)
ax.grid(True, which="minor", linestyle=':', alpha=0.4)

ax.legend(frameon=False)

# Clean spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('runtime_scaling_loglog_neurips.pdf', format='pdf', bbox_inches='tight')
plt.show()
