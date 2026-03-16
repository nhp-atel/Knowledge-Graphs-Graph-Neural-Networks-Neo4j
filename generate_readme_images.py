"""Generate images for the README introduction."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import networkx as nx

# ============================================================
# Image 1: Knowledge Graph Visualization
# ============================================================

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')

G = nx.DiGraph()

# Nodes with types
nodes = {
    'Louisville\nHub':       {'type': 'hub',       'pos': (0.5, 0.75)},
    'Chicago\nHub':          {'type': 'hub',       'pos': (0.2, 0.55)},
    'Dallas\nHub':           {'type': 'hub',       'pos': (0.8, 0.55)},
    'Anchorage\nAir Gateway':{'type': 'gateway',   'pos': (0.05, 0.3)},
    'Ontario\nAir Gateway':  {'type': 'gateway',   'pos': (0.95, 0.3)},
    'Nashville\nDepot':      {'type': 'depot',     'pos': (0.35, 0.35)},
    'Memphis\nDepot':        {'type': 'depot',     'pos': (0.65, 0.35)},
    'PKG-1247':              {'type': 'package',   'pos': (0.15, 0.1)},
    'PKG-3891':              {'type': 'package',   'pos': (0.5, 0.1)},
    'PKG-7702':              {'type': 'package',   'pos': (0.85, 0.1)},
}

edges = [
    ('Louisville\nHub', 'Chicago\nHub', 'ROUTES_TO\n265 mi'),
    ('Louisville\nHub', 'Dallas\nHub', 'ROUTES_TO\n726 mi'),
    ('Louisville\nHub', 'Nashville\nDepot', 'ROUTES_TO\n175 mi'),
    ('Louisville\nHub', 'Anchorage\nAir Gateway', 'FLIES_TO'),
    ('Dallas\nHub', 'Memphis\nDepot', 'ROUTES_TO\n450 mi'),
    ('Dallas\nHub', 'Ontario\nAir Gateway', 'FLIES_TO'),
    ('Chicago\nHub', 'Anchorage\nAir Gateway', 'FLIES_TO'),
    ('PKG-1247', 'Chicago\nHub', 'LOCATED_AT'),
    ('PKG-3891', 'Nashville\nDepot', 'LOCATED_AT'),
    ('PKG-7702', 'Ontario\nAir Gateway', 'LOCATED_AT'),
]

for node, attrs in nodes.items():
    G.add_node(node, **attrs)
for src, dst, label in edges:
    G.add_edge(src, dst, label=label)

pos = {n: attrs['pos'] for n, attrs in nodes.items()}

type_styles = {
    'hub':     {'color': '#e74c3c', 'size': 2800, 'shape': 'o'},
    'gateway': {'color': '#3498db', 'size': 2200, 'shape': 'o'},
    'depot':   {'color': '#2ecc71', 'size': 1800, 'shape': 'o'},
    'package': {'color': '#f39c12', 'size': 1400, 'shape': 's'},
}

# Draw edges
for src, dst, label in edges:
    x1, y1 = pos[src]
    x2, y2 = pos[dst]
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#555555',
                                lw=1.8, connectionstyle='arc3,rad=0.1'))
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    # Offset label slightly
    offset_x = (y2 - y1) * 0.04
    offset_y = -(x2 - x1) * 0.04

# Draw nodes by type
for ntype, style in type_styles.items():
    type_nodes = [n for n, a in nodes.items() if a['type'] == ntype]
    nx.draw_networkx_nodes(G, pos, nodelist=type_nodes, ax=ax,
                           node_color=style['color'], node_size=style['size'],
                           node_shape=style['shape'], alpha=0.95,
                           edgecolors='white', linewidths=2)

# Draw labels
for node, (x, y) in pos.items():
    ax.text(x, y, node, fontsize=8, fontweight='bold', color='white',
            ha='center', va='center', family='monospace')

# Edge labels (selected)
edge_labels_to_show = [
    ('Louisville\nHub', 'Chicago\nHub', '265 mi'),
    ('Louisville\nHub', 'Dallas\nHub', '726 mi'),
    ('Louisville\nHub', 'Nashville\nDepot', '175 mi'),
    ('Dallas\nHub', 'Memphis\nDepot', '450 mi'),
]
for src, dst, label in edge_labels_to_show:
    x1, y1 = pos[src]
    x2, y2 = pos[dst]
    mx, my = (x1 + x2) / 2 + 0.02, (y1 + y2) / 2 + 0.02
    ax.text(mx, my, label, fontsize=7, color='#aaaaaa', ha='center',
            va='center', family='monospace',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#0d1117',
                      edgecolor='#333333', alpha=0.9))

# Legend
legend_elements = [
    mpatches.Patch(facecolor='#e74c3c', edgecolor='white', label='Hub'),
    mpatches.Patch(facecolor='#3498db', edgecolor='white', label='Air Gateway'),
    mpatches.Patch(facecolor='#2ecc71', edgecolor='white', label='Depot'),
    mpatches.Patch(facecolor='#f39c12', edgecolor='white', label='Package'),
]
legend = ax.legend(handles=legend_elements, loc='lower left', fontsize=10,
                   facecolor='#161b22', edgecolor='#333333', labelcolor='white',
                   framealpha=0.95)

ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.05, 0.9)
ax.set_title('Logistics Knowledge Graph', fontsize=18, fontweight='bold',
             color='white', pad=15)
ax.axis('off')

plt.tight_layout()
plt.savefig('images/knowledge_graph.png', dpi=150, bbox_inches='tight',
            facecolor='#0d1117', edgecolor='none')
plt.close()
print('Saved images/knowledge_graph.png')


# ============================================================
# Image 2: Loss Surface with Bezier Curve Connecting Optima
# ============================================================

fig = plt.figure(figsize=(12, 7))
fig.patch.set_facecolor('#0d1117')
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('#0d1117')

# Generate a loss surface with two basins (optima)
x = np.linspace(-3, 3, 200)
y = np.linspace(-3, 3, 200)
X, Y = np.meshgrid(x, y)

# Two-basin loss surface
Z = (0.4 * ((X - 1.2)**2 * (Y - 0.8)**2 + 0.3 * (X - 1.2)**2 + 0.3 * (Y - 0.8)**2)
     + 0.4 * ((X + 1.2)**2 * (Y + 0.5)**2 + 0.3 * (X + 1.2)**2 + 0.3 * (Y + 0.5)**2)
     - 0.8 * np.exp(-((X - 1.2)**2 + (Y - 0.8)**2) / 0.8)
     - 0.8 * np.exp(-((X + 1.2)**2 + (Y + 0.5)**2) / 0.8)
     + 0.15 * np.sin(2 * X) * np.cos(2 * Y)
     + 0.5)

# Clip for better visualization
Z = np.clip(Z, 0, 6)

# Plot surface
ax.plot_surface(X, Y, Z, cmap='inferno', alpha=0.6, rstride=4, cstride=4,
                edgecolor='none', antialiased=True)

# Optima locations
opt1 = np.array([1.2, 0.8])
opt2 = np.array([-1.2, -0.5])

# Evaluate Z at optima
def eval_z(px, py):
    return (0.4 * ((px - 1.2)**2 * (py - 0.8)**2 + 0.3 * (px - 1.2)**2 + 0.3 * (py - 0.8)**2)
            + 0.4 * ((px + 1.2)**2 * (py + 0.5)**2 + 0.3 * (px + 1.2)**2 + 0.3 * (py + 0.5)**2)
            - 0.8 * np.exp(-((px - 1.2)**2 + (py - 0.8)**2) / 0.8)
            - 0.8 * np.exp(-((px + 1.2)**2 + (py + 0.5)**2) / 0.8)
            + 0.15 * np.sin(2 * px) * np.cos(2 * py)
            + 0.5)

z1 = eval_z(opt1[0], opt1[1])
z2 = eval_z(opt2[0], opt2[1])

# Plot optima as stars
ax.scatter([opt1[0]], [opt1[1]], [z1], color='#00ff88', s=200, marker='*',
           zorder=10, edgecolors='white', linewidths=1.5)
ax.scatter([opt2[0]], [opt2[1]], [z2], color='#00ff88', s=200, marker='*',
           zorder=10, edgecolors='white', linewidths=1.5)

# Bezier curve connecting the two optima (quadratic Bezier)
# Control point is above and between the two optima
control = np.array([0.0, 0.8])

t = np.linspace(0, 1, 100)
bezier_x = (1-t)**2 * opt1[0] + 2*(1-t)*t * control[0] + t**2 * opt2[0]
bezier_y = (1-t)**2 * opt1[1] + 2*(1-t)*t * control[1] + t**2 * opt2[1]
bezier_z = np.array([eval_z(bx, by) for bx, by in zip(bezier_x, bezier_y)])

# Add a small offset above the surface so the curve is visible
bezier_z_plot = bezier_z + 0.05

ax.plot(bezier_x, bezier_y, bezier_z_plot, color='#00ff88', linewidth=3.5,
        zorder=10, alpha=0.95)

# Dashed projection of curve onto the base
ax.plot(bezier_x, bezier_y, np.zeros_like(bezier_z) - 0.1,
        color='#00ff88', linewidth=1.5, linestyle='--', alpha=0.4)

# Labels
ax.text(opt1[0] + 0.15, opt1[1] + 0.2, z1 + 0.5,
        r'$\theta_1^*$' + '\n(Optimum 1)', color='white', fontsize=11,
        fontweight='bold', ha='center')
ax.text(opt2[0] - 0.15, opt2[1] - 0.3, z2 + 0.5,
        r'$\theta_2^*$' + '\n(Optimum 2)', color='white', fontsize=11,
        fontweight='bold', ha='center')

# Curve label
mid_t = 0.5
mid_x = (1-mid_t)**2 * opt1[0] + 2*(1-mid_t)*mid_t * control[0] + mid_t**2 * opt2[0]
mid_y = (1-mid_t)**2 * opt1[1] + 2*(1-mid_t)*mid_t * control[1] + mid_t**2 * opt2[1]
mid_z = eval_z(mid_x, mid_y)
ax.text(mid_x + 0.3, mid_y + 0.5, mid_z + 1.0,
        'Low-loss\npathway', color='#00ff88', fontsize=11,
        fontweight='bold', ha='center', style='italic')

# Axis styling
ax.set_xlabel('Weight dimension 1', fontsize=10, color='#aaaaaa', labelpad=8)
ax.set_ylabel('Weight dimension 2', fontsize=10, color='#aaaaaa', labelpad=8)
ax.set_zlabel('Loss', fontsize=10, color='#aaaaaa', labelpad=8)

ax.set_title('Loss Surface: Optima Connected by a Low-Loss Bezier Curve\n'
             '(Fast Geometric Ensembling, Garipov et al.)',
             fontsize=14, fontweight='bold', color='white', pad=20)

# Style the axes
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('#333333')
ax.yaxis.pane.set_edgecolor('#333333')
ax.zaxis.pane.set_edgecolor('#333333')
ax.tick_params(colors='#666666', labelsize=8)
ax.xaxis.line.set_color('#333333')
ax.yaxis.line.set_color('#333333')
ax.zaxis.line.set_color('#333333')

ax.view_init(elev=30, azim=-50)

plt.tight_layout()
plt.savefig('images/loss_surface_bezier.png', dpi=150, bbox_inches='tight',
            facecolor='#0d1117', edgecolor='none')
plt.close()
print('Saved images/loss_surface_bezier.png')
