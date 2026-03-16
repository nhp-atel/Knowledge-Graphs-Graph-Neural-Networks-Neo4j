"""Generate an animated GIF showing how a GNN works step by step."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
import networkx as nx

np.random.seed(42)

# ==============================================================
# Build the friend network (same as before)
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
features_list = ['music', 'sports', 'cooking', 'gaming']
H0 = np.array([[friends.nodes[name][f] for f in features_list] for name in names], dtype=float)
A = nx.adjacency_matrix(friends, nodelist=names).toarray().astype(float)
A_tilde = A + np.eye(n)
D_inv_sqrt = np.diag(1.0 / np.sqrt(A_tilde.sum(axis=1)))
A_hat = D_inv_sqrt @ A_tilde @ D_inv_sqrt

H1 = A_hat @ H0
H2 = A_hat @ H1

# Recommendations
existing_edges = set(friends.edges())
existing_edges.update((v, u) for u, v in friends.edges())

recommendations = []
for i in range(n):
    for j in range(i + 1, n):
        if (names[i], names[j]) not in existing_edges:
            score = H2[i] @ H2[j]
            recommendations.append((names[i], names[j], round(score, 3)))

recommendations.sort(key=lambda x: -x[2])
top3 = recommendations[:3]


# ==============================================================
# Animation
# ==============================================================

BG = '#0d1117'
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

# Total frames: we define phases
# Phase 0 (frames 0-14):   Title card
# Phase 1 (frames 15-34):  Show the network + interests
# Phase 2 (frames 35-54):  Message passing layer 1 (highlight Alice's neighbors)
# Phase 3 (frames 55-74):  Message passing layer 2 (2-hop info)
# Phase 4 (frames 75-94):  Show predictions
# Phase 5 (frames 95-114): Final summary
TOTAL_FRAMES = 115
FPS = 5  # slow enough to read


def get_interest_str(name):
    p = people[name]
    tags = [k for k, v in p.items() if v == 1]
    return ', '.join(tags)


def draw_network(ax, node_colors, edge_colors=None, edge_widths=None,
                 extra_edges=None, title='', subtitle='', annotation=''):
    ax.clear()
    ax.set_facecolor(BG)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis('off')

    if edge_colors is None:
        edge_colors = ['#444444'] * friends.number_of_edges()
    if edge_widths is None:
        edge_widths = [2.0] * friends.number_of_edges()

    # Draw existing edges
    nx.draw_networkx_edges(friends, pos, ax=ax, edge_color=edge_colors,
                           width=edge_widths, alpha=0.7)

    # Draw extra edges (predictions)
    if extra_edges:
        for p1, p2, score in extra_edges:
            x1, y1 = pos[p1]
            x2, y2 = pos[p2]
            ax.plot([x1, x2], [y1, y2], color='#e74c3c', linewidth=3,
                    linestyle='--', alpha=0.9, zorder=0)
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my + 0.08, f'{score:.2f}', fontsize=10,
                    color='#e74c3c', ha='center', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=BG,
                              edgecolor='#e74c3c', alpha=0.9))

    # Draw nodes
    nx.draw_networkx_nodes(friends, pos, ax=ax, node_color=node_colors,
                           node_size=1400, edgecolors='white', linewidths=2.5)
    nx.draw_networkx_labels(friends, pos, ax=ax, font_size=11,
                            font_weight='bold', font_color='white')

    # Title
    if title:
        ax.text(0.5, 1.08, title, transform=ax.transAxes, fontsize=18,
                fontweight='bold', color='white', ha='center', va='top')
    if subtitle:
        ax.text(0.5, 1.01, subtitle, transform=ax.transAxes, fontsize=12,
                color='#aaaaaa', ha='center', va='top')

    # Annotation box (bottom left)
    if annotation:
        ax.text(0.02, 0.02, annotation, transform=ax.transAxes, fontsize=10,
                color='#4ECDC4', fontfamily='monospace', verticalalignment='bottom',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#161b22',
                          edgecolor='#333333', alpha=0.95))


def animate(frame):
    default_color = '#4ECDC4'
    alice_i = name_to_i['Alice']

    # --- Phase 0: Title card ---
    if frame < 15:
        ax.clear()
        ax.set_facecolor(BG)
        ax.axis('off')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.text(0.5, 0.6, 'How a Graph Neural Network Works',
                fontsize=24, fontweight='bold', color='white',
                ha='center', va='center')
        ax.text(0.5, 0.45, 'A step-by-step simulation with 8 friends',
                fontsize=14, color='#aaaaaa', ha='center', va='center')
        ax.text(0.5, 0.3, 'Can a GNN predict who should become friends?',
                fontsize=13, color='#4ECDC4', ha='center', va='center',
                style='italic')
        return

    # --- Phase 1: Show network + interests ---
    elif frame < 35:
        colors = [default_color] * n
        interests_text = '\n'.join([f'{name}: {get_interest_str(name)}'
                                     for name in names])
        draw_network(ax, colors,
                     title='Step 1: The Friend Network',
                     subtitle='8 people, 9 friendships, 4 interests each',
                     annotation=interests_text)
        return

    # --- Phase 2: Message passing layer 1 ---
    elif frame < 55:
        sub_frame = frame - 35
        colors = [default_color] * n

        # Highlight Alice
        colors[alice_i] = '#FFD700'

        # Highlight Alice's direct friends
        alice_neighbors = [name_to_i[nb] for nb in friends.neighbors('Alice')]
        for ni in alice_neighbors:
            colors[ni] = '#FF6B6B'

        # Edges from Alice highlighted
        edge_list = list(friends.edges())
        edge_colors = []
        edge_widths = []
        for u, v in edge_list:
            if u == 'Alice' or v == 'Alice':
                edge_colors.append('#FFD700')
                edge_widths.append(4.0)
            else:
                edge_colors.append('#333333')
                edge_widths.append(1.5)

        ai = alice_i
        vals = H1[ai]
        ann = (f"Alice collects from friends:\n"
               f"  Bob:  music, sports\n"
               f"  Eve:  music, cooking\n"
               f"\n"
               f"After Layer 1:\n"
               f"  Music={vals[0]:.2f}  Sports={vals[1]:.2f}\n"
               f"  Cooking={vals[2]:.2f}  Gaming={vals[3]:.2f}")

        draw_network(ax, colors, edge_colors, edge_widths,
                     title='Step 2: Message Passing (Layer 1)',
                     subtitle='Each person collects information from direct friends',
                     annotation=ann)
        return

    # --- Phase 3: Message passing layer 2 ---
    elif frame < 75:
        # Color all nodes by how much they changed from original
        diffs = np.linalg.norm(H2 - H0, axis=1)
        diffs_norm = diffs / diffs.max()
        colors = [plt.cm.YlOrRd(0.15 + 0.75 * d) for d in diffs_norm]

        # Still highlight Alice
        colors[alice_i] = '#FFD700'

        ai = alice_i
        vals = H2[ai]
        ann = (f"After Layer 2, Alice also knows about\n"
               f"friends-of-friends: Charlie, Grace, Diana\n"
               f"\n"
               f"Alice's final representation:\n"
               f"  Music={vals[0]:.2f}  Sports={vals[1]:.2f}\n"
               f"  Cooking={vals[2]:.2f}  Gaming={vals[3]:.2f}\n"
               f"\n"
               f"Node color = how much each person\n"
               f"changed from their original interests")

        draw_network(ax, colors,
                     title='Step 2: Message Passing (Layer 2)',
                     subtitle='Now includes friends-of-friends information',
                     annotation=ann)
        return

    # --- Phase 4: Show predictions ---
    elif frame < 95:
        colors = [default_color] * n
        # Highlight the predicted pairs
        for p1, p2, _ in top3:
            colors[name_to_i[p1]] = '#FF6B6B'
            colors[name_to_i[p2]] = '#FF6B6B'

        ann = (f"Top 3 predictions:\n"
               f"  1. {top3[0][0]} + {top3[0][1]}  (score: {top3[0][2]})\n"
               f"  2. {top3[1][0]} + {top3[1][1]}  (score: {top3[1][2]})\n"
               f"  3. {top3[2][0]} + {top3[2][1]}  (score: {top3[2][2]})\n"
               f"\n"
               f"All have mutual friends and\n"
               f"overlapping interests -- the GNN\n"
               f"discovered this on its own.")

        draw_network(ax, colors, extra_edges=top3,
                     title='Step 3: Predict New Friendships',
                     subtitle='Red dashed = GNN recommendation (dot-product similarity)',
                     annotation=ann)
        return

    # --- Phase 5: Summary ---
    else:
        ax.clear()
        ax.set_facecolor(BG)
        ax.axis('off')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.text(0.5, 0.72, 'That is a Graph Neural Network.',
                fontsize=22, fontweight='bold', color='white',
                ha='center', va='center')
        ax.text(0.5, 0.58,
                '1.  Represent entities as feature vectors\n'
                '2.  Let information flow through the network\n'
                '3.  Measure who ends up similar',
                fontsize=14, color='#aaaaaa', ha='center', va='center',
                fontfamily='monospace', linespacing=1.8)
        ax.text(0.5, 0.32,
                'The rest of this guide applies the same idea\n'
                'to logistics networks, knowledge graphs,\n'
                'and real-world ML pipelines.',
                fontsize=13, color='#4ECDC4', ha='center', va='center',
                style='italic', linespacing=1.6)
        return


print(f'Generating {TOTAL_FRAMES} frames at {FPS} fps...')
anim = FuncAnimation(fig, animate, frames=TOTAL_FRAMES, interval=1000//FPS)
anim.save('images/gnn_simulation.gif', writer=PillowWriter(fps=FPS),
          savefig_kwargs={'facecolor': BG, 'edgecolor': 'none'})
plt.close()
print('Saved images/gnn_simulation.gif')

# Print file size
import os
size_mb = os.path.getsize('images/gnn_simulation.gif') / (1024 * 1024)
print(f'File size: {size_mb:.1f} MB')
