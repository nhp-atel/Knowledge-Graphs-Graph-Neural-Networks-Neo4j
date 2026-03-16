"""Generate GNN simulation images for the README."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import networkx as nx
import pandas as pd

np.random.seed(42)

# ==============================================================
# Build the friend network
# ==============================================================

friends = nx.Graph()

people = {
    'Alice':   {'music': 1, 'sports': 0, 'cooking': 1, 'gaming': 0},
    'Bob':     {'music': 1, 'sports': 1, 'cooking': 0, 'gaming': 0},
    'Charlie': {'music': 0, 'sports': 1, 'cooking': 0, 'gaming': 1},
    'Diana':   {'music': 0, 'sports': 0, 'cooking': 1, 'gaming': 1},
    'Eve':     {'music': 1, 'sports': 0, 'cooking': 1, 'gaming': 0},
    'Frank':   {'music': 0, 'sports': 1, 'cooking': 0, 'gaming': 1},
    'Grace':   {'music': 1, 'sports': 1, 'cooking': 0, 'gaming': 0},
    'Hank':    {'music': 0, 'sports': 0, 'cooking': 1, 'gaming': 1},
}

for name, interests in people.items():
    friends.add_node(name, **interests)

friendships = [
    ('Alice', 'Bob'), ('Alice', 'Eve'), ('Bob', 'Charlie'),
    ('Bob', 'Grace'), ('Charlie', 'Frank'), ('Diana', 'Hank'),
    ('Diana', 'Eve'), ('Frank', 'Hank'), ('Grace', 'Charlie'),
]
friends.add_edges_from(friendships)

pos = nx.spring_layout(friends, seed=42, k=2.0)
names = sorted(friends.nodes())
n = len(names)
name_to_i = {name: i for i, name in enumerate(names)}

# Feature matrix and normalized adjacency
features = ['music', 'sports', 'cooking', 'gaming']
H0 = np.array([[friends.nodes[name][f] for f in features] for name in names], dtype=float)
A = nx.adjacency_matrix(friends, nodelist=names).toarray().astype(float)
A_tilde = A + np.eye(n)
D_inv_sqrt = np.diag(1.0 / np.sqrt(A_tilde.sum(axis=1)))
A_hat = D_inv_sqrt @ A_tilde @ D_inv_sqrt

H1 = A_hat @ H0
H2 = A_hat @ H1

# Get recommendations
existing_edges = set(friends.edges())
existing_edges.update((v, u) for u, v in friends.edges())

recommendations = []
for i in range(n):
    for j in range(i + 1, n):
        if (names[i], names[j]) not in existing_edges:
            score = H2[i] @ H2[j]
            mutual = len(set(friends.neighbors(names[i])) & set(friends.neighbors(names[j])))
            recommendations.append({
                'Person 1': names[i], 'Person 2': names[j],
                'GNN Score': round(score, 3), 'Mutual Friends': mutual,
            })

rec_df = pd.DataFrame(recommendations).sort_values('GNN Score', ascending=False)

# ==============================================================
# Image 1: The friend network with interests table
# ==============================================================

fig, (ax_graph, ax_table) = plt.subplots(1, 2, figsize=(16, 7),
                                          gridspec_kw={'width_ratios': [1.2, 1]})
fig.patch.set_facecolor('#0d1117')
ax_graph.set_facecolor('#0d1117')
ax_table.set_facecolor('#0d1117')

nx.draw(friends, pos, ax=ax_graph, with_labels=True,
        node_color='#4ECDC4', node_size=1400, font_size=12,
        font_weight='bold', edge_color='#555555', width=2.5,
        edgecolors='white', linewidths=2.5)
ax_graph.set_title('Friend Network', fontsize=16, fontweight='bold', color='white')

# Interests table
interest_data = [[people[name][f] for f in features] for name in sorted(people.keys())]
table = ax_table.table(
    cellText=interest_data,
    rowLabels=sorted(people.keys()),
    colLabels=[f.capitalize() for f in features],
    cellLoc='center',
    loc='center',
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.0, 1.8)

# Style the table
for key, cell in table.get_celld().items():
    cell.set_edgecolor('#333333')
    if key[0] == 0:  # header
        cell.set_facecolor('#2d333b')
        cell.set_text_props(color='white', fontweight='bold')
    elif key[1] == -1:  # row labels
        cell.set_facecolor('#2d333b')
        cell.set_text_props(color='#4ECDC4', fontweight='bold')
    else:
        val = interest_data[key[0]-1][key[1]]
        cell.set_facecolor('#1a2332' if val == 0 else '#1a3a2a')
        cell.set_text_props(color='#666666' if val == 0 else '#4ECDC4')

ax_table.set_title("Each Person's Interests", fontsize=16, fontweight='bold', color='white')
ax_table.axis('off')

plt.suptitle('Step 1: Build a Social Graph',
             fontsize=18, fontweight='bold', color='white', y=1.02)
plt.tight_layout()
plt.savefig('images/sim_step1_network.png', dpi=150, bbox_inches='tight',
            facecolor='#0d1117', edgecolor='none')
plt.close()
print('Saved images/sim_step1_network.png')


# ==============================================================
# Image 2: Message passing visualization
# ==============================================================

fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.patch.set_facecolor('#0d1117')

titles = ['Before GNN\n(Raw Interests)', 'After Layer 1\n(Friends\' Info Mixed In)',
          'After Layer 2\n(Friends-of-Friends)']
matrices = [H0, H1, H2]

for ax, title, H in zip(axes, titles, matrices):
    ax.set_facecolor('#0d1117')

    # Color nodes by how much their representation has changed
    if H is H0:
        node_colors = ['#4ECDC4'] * n
    else:
        # Color intensity by how different from original
        diffs = np.linalg.norm(H - H0, axis=1)
        diffs_norm = diffs / diffs.max()
        node_colors = [plt.cm.YlOrRd(0.2 + 0.7 * d) for d in diffs_norm]

    nx.draw(friends, pos, ax=ax, with_labels=True,
            node_color=node_colors, node_size=1200, font_size=10,
            font_weight='bold', edge_color='#444444', width=2,
            edgecolors='white', linewidths=2)

    ax.set_title(title, fontsize=14, fontweight='bold', color='white')

    # Add Alice's values as annotation
    ai = name_to_i['Alice']
    alice_vals = H[ai]
    text = f"Alice: M={alice_vals[0]:.2f} S={alice_vals[1]:.2f}\n       C={alice_vals[2]:.2f} G={alice_vals[3]:.2f}"
    ax.text(0.02, 0.02, text, transform=ax.transAxes, fontsize=9,
            color='#4ECDC4', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#161b22',
                      edgecolor='#333333', alpha=0.95),
            verticalalignment='bottom')

plt.suptitle('Step 2: Message Passing — Watch Information Flow Through the Network',
             fontsize=16, fontweight='bold', color='white', y=1.02)
plt.tight_layout()
plt.savefig('images/sim_step2_message_passing.png', dpi=150, bbox_inches='tight',
            facecolor='#0d1117', edgecolor='none')
plt.close()
print('Saved images/sim_step2_message_passing.png')


# ==============================================================
# Image 3: Predicted friendships
# ==============================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7),
                                 gridspec_kw={'width_ratios': [1.2, 1]})
fig.patch.set_facecolor('#0d1117')
ax1.set_facecolor('#0d1117')
ax2.set_facecolor('#0d1117')

# Left: network with predicted edges
nx.draw(friends, pos, ax=ax1, with_labels=True,
        node_color='#4ECDC4', node_size=1200, font_size=11,
        font_weight='bold', edge_color='#444444', width=2,
        edgecolors='white', linewidths=2)

# Draw top 3 predictions as dashed lines
top3 = rec_df.head(3)
for _, row in top3.iterrows():
    p1, p2 = row['Person 1'], row['Person 2']
    x1, y1 = pos[p1]
    x2, y2 = pos[p2]
    ax1.plot([x1, x2], [y1, y2], color='#e74c3c', linewidth=3,
             linestyle='--', alpha=0.9, zorder=0)
    # Score label at midpoint
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    ax1.text(mx, my + 0.04, f'{row["GNN Score"]:.2f}', fontsize=9,
             color='#e74c3c', ha='center', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='#0d1117',
                       edgecolor='#e74c3c', alpha=0.9))

legend_elements = [
    Line2D([0], [0], color='#444444', linewidth=2, label='Existing friendship'),
    Line2D([0], [0], color='#e74c3c', linewidth=3, linestyle='--', label='GNN recommendation'),
]
ax1.legend(handles=legend_elements, loc='lower center', fontsize=10,
           facecolor='#161b22', edgecolor='#333333', labelcolor='white')
ax1.set_title('Top 3 Friend Recommendations', fontsize=15, fontweight='bold', color='white')

# Right: recommendation table (top 8)
top8 = rec_df.head(8)
table_data = [[row['Person 1'], row['Person 2'], f"{row['GNN Score']:.3f}", str(row['Mutual Friends'])]
              for _, row in top8.iterrows()]
table = ax2.table(
    cellText=table_data,
    colLabels=['Person 1', 'Person 2', 'GNN Score', 'Mutual\nFriends'],
    cellLoc='center',
    loc='center',
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.0, 1.8)

for key, cell in table.get_celld().items():
    cell.set_edgecolor('#333333')
    if key[0] == 0:
        cell.set_facecolor('#2d333b')
        cell.set_text_props(color='white', fontweight='bold')
    elif key[0] <= 3:  # top 3 highlighted
        cell.set_facecolor('#2a1a1a')
        cell.set_text_props(color='#e74c3c', fontweight='bold')
    else:
        cell.set_facecolor('#161b22')
        cell.set_text_props(color='#aaaaaa')

ax2.set_title('Ranked by GNN Similarity', fontsize=15, fontweight='bold', color='white')
ax2.axis('off')

plt.suptitle('Step 3: Predict New Friendships — Who Should Meet?',
             fontsize=17, fontweight='bold', color='white', y=1.02)
plt.tight_layout()
plt.savefig('images/sim_step3_predictions.png', dpi=150, bbox_inches='tight',
            facecolor='#0d1117', edgecolor='none')
plt.close()
print('Saved images/sim_step3_predictions.png')
