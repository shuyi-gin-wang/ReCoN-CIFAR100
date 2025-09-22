from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple, Union
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import torch
from matplotlib.colors import to_rgba

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from recon_cifar import CIFARReCoNBuilder
from recon import ReCoN, SUB, SUR, POR, RET, GEN

from train_common import get_loaders, load_model_from_pt, CLASS_TO_NAME, SUPER_TO_NAME

EDGE_STYLE = {
    SUB: {"label": "SUB", "color": "tab:blue"},
    SUR: {"label": "SUR", "color": "tab:green"},
    POR: {"label": "POR", "color": "tab:red"},
    RET: {"label": "RET", "color": "tab:orange"},
    GEN: {"label": "GEN", "color": "tab:gray"},
}

@dataclass
class Edge:
    src: int
    dst: int
    weight: float
    kind: int  # one of SUB/SUR/POR/RET/GEN


def _iter_matrix_edges(mat: np.ndarray, kind: int, eps: float = 0.0) -> Iterable[Edge]:
    if mat.ndim != 2:
        return []
    rows, cols = np.nonzero(np.abs(mat) > eps)
    for i, j in zip(rows.tolist(), cols.tolist()):
        yield Edge(i, j, float(mat[i, j]), kind)


def recon_to_edges(recon, include_gen=True, show_all_gen=False, eps: float = 1e-12) -> Iterable[Edge]:
    yield from _iter_matrix_edges(recon.w_sub, SUB, eps)
    yield from _iter_matrix_edges(recon.w_sur, SUR, eps)
    yield from _iter_matrix_edges(recon.w_por, POR, eps)
    yield from _iter_matrix_edges(recon.w_ret, RET, eps)

    if include_gen:
        w_gen = getattr(recon, "w_gen", None)
        if w_gen is not None and w_gen.ndim == 1:
            for i, w in enumerate(w_gen.tolist()):
                if show_all_gen or (abs(w - 1.0) > eps):
                    yield Edge(i, i, float(w), GEN)


def recon_to_digraph(recon, include_gen=True, show_all_gen=False, eps: float = 1e-12) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from(range(recon.num))
    for e in recon_to_edges(recon, include_gen=include_gen, show_all_gen=show_all_gen, eps=eps):
        G.add_edge(e.src, e.dst, weight=e.weight, kind=e.kind)
    return G


def _edge_artist_by_kind(G: nx.DiGraph, kind: int) -> Tuple[list, list, list]:
    edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("kind") == kind]
    weights = [abs(G[u][v]["weight"]) for u, v in edges]
    styles = ["dashed" if G[u][v]["weight"] < 0 else "solid" for u, v in edges]
    return edges, weights, styles


def draw_recon_graph(
    G: nx.DiGraph,
    recon_builder: CIFARReCoNBuilder,
    pos: Optional[Dict[int, Tuple[float, float]]] = None,
    title: str = "ReCoN",
    font_size: int = 10,
    k: Optional[float] = None,
    seed: int = 13,
    arrowsize: int = 12,
    edge_width_scale: float = 2.0,
    legend: bool = True,
    sample_image: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    inset_bounds: Tuple[float, float, float, float] = (0.8, 0.8, 0.18, 0.18),  # (x0,y0,w,h) in axes coords
    cifar100_norm: Tuple[Tuple[float,float,float], Tuple[float,float,float]] = (
        (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    ),
    new_node_indices = None
):
    if pos is None:
        n = G.number_of_nodes()
        if k is None:
            k = 1 / math.sqrt(max(n, 1))
        pos = nx.spring_layout(G, k=k, seed=seed)

    fig = plt.figure(figsize=(9, 7))
    ax = plt.gca()

    # (existing node + edge drawing code unchanged) ...
    # --- Nodes
    recon_builder.class_terminal_netid_mapping.keys() 
    recon_builder.superclass_terminal_netid_mapping.keys()
    
    labels = None
    def node_labels(n):
        if n in recon_builder.class_terminal_netid_mapping:
            return CLASS_TO_NAME[recon_builder.class_terminal_netid_mapping[n]]
        if n in recon_builder.superclass_terminal_netid_mapping:
            return SUPER_TO_NAME[recon_builder.superclass_terminal_netid_mapping[n]]
        return n
    labels = {n: node_labels(n) for n in G.nodes()}

    default_c = "#48b6ff"
    class_c   = "#a090ff"
    super_c   = "#ffea28"

    default_s = 200
    class_s   = 500
    super_s   = 800 

    class_nodes = set(recon_builder.class_terminal_netid_mapping)
    super_nodes = set(recon_builder.superclass_terminal_netid_mapping)
    
    color_map = {n: super_c for n in super_nodes}
    color_map.update({n: class_c for n in class_nodes})

    size_map = {n: super_s for n in super_nodes}
    size_map.update({n: class_s for n in class_nodes})

    colors = [color_map.get(n, default_c) for n in G.nodes()]
    sizes  = [size_map.get(n, default_s)  for n in G.nodes()]

    default_a = 0.4
    new_node_a  = 1

    alphas = [new_node_a if n in new_node_indices else default_a for n in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_size=sizes, ax=ax, node_color=colors, alpha=alphas)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=font_size, ax=ax)

    # --- Edges (unchanged)
    new_nodes = set(new_node_indices or [])

    def edge_is_new(u, v):
        # treat an edge as "new" if it touches any new node
        return (u in new_nodes) or (v in new_nodes)

    handles = []
    for kind in [SUB, SUR, POR, RET, GEN]:
        edges, weights, styles = _edge_artist_by_kind(G, kind)
        if not edges:
            continue

        widths = [max(0.5, edge_width_scale * w) for w in weights]
        base_c = EDGE_STYLE[kind]["color"]
        label  = EDGE_STYLE[kind]["label"]

        edge_colors = [
            to_rgba(base_c, 1.0 if edge_is_new(u, v) else 0.4)
            for (u, v) in edges
        ]

        # Split by style so dashed negatives draw correctly
        solid_mask  = [s == "solid"  for s in styles]
        dashed_mask = [s == "dashed" for s in styles]

        solid_edges   = [e for e, m in zip(edges, solid_mask) if m]
        solid_widths  = [w for w, m in zip(widths, solid_mask) if m]
        solid_colors  = [c for c, m in zip(edge_colors, solid_mask) if m]

        dashed_edges  = [e for e, m in zip(edges, dashed_mask) if m]
        dashed_widths = [w for w, m in zip(widths, dashed_mask) if m]
        dashed_colors = [c for c, m in zip(edge_colors, dashed_mask) if m]

        if solid_edges:
            nx.draw_networkx_edges(
                G, pos,
                edgelist=solid_edges,
                width=solid_widths,
                arrowstyle="-|>",
                arrows=True,
                arrowsize=arrowsize,
                edge_color=solid_colors,   # per-edge RGBA (controls alpha)
                style="solid",
                connectionstyle="arc3,rad=0.05",
                ax=ax,
            )

        if dashed_edges:
            nx.draw_networkx_edges(
                G, pos,
                edgelist=dashed_edges,
                width=dashed_widths,
                arrowstyle="-|>",
                arrows=True,
                arrowsize=arrowsize,
                edge_color=dashed_colors,  # per-edge RGBA (controls alpha)
                style="dashed",
                connectionstyle="arc3,rad=0.05",
                ax=ax,
            )

        handles.append(plt.Line2D([0], [0], color=EDGE_STYLE[kind]["color"], lw=3, label=EDGE_STYLE[kind]["label"]))
    if legend and handles:
        ax.legend(handles=handles, loc="upper left", frameon=False)

    if sample_image is not None:
        mean, std = cifar100_norm
        img = _to_numpy_image(sample_image, mean=mean, std=std)
        # Put a neat inset in the upper-right corner
        # You can tweak the bounds via the function argument
        x0, y0, w, h = inset_bounds
        inset_ax: Axes = ax.inset_axes([x0, y0, w, h])
        inset_ax.imshow(img)
        inset_ax.set_xticks([]); inset_ax.set_yticks([])
        inset_ax.set_title("Input", fontsize=9, pad=2)

        # optional: add a thin border around the inset
        for spine in inset_ax.spines.values():
            spine.set_linewidth(1.0)

    ax.set_title(title)
    ax.axis("off")
    if legend and handles:
        ax.legend(handles=handles, loc="upper left", frameon=False)
    fig.tight_layout()
    return fig, pos

def visualize_recon(
    recon_builder,
    title: str = "ReCoN",
    save_png: Optional[str] = None,
    include_gen: bool = True,
    show_all_gen: bool = False,
    eps: float = 1e-12,
    sample_image: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    new_node_indices = None
):
    G = recon_to_digraph(recon_builder.net, include_gen=include_gen, show_all_gen=show_all_gen, eps=eps)
    fig, pos = draw_recon_graph(G, recon_builder, title=title, sample_image=sample_image, new_node_indices=new_node_indices)
    if save_png:
        fig.savefig(save_png, dpi=200)
        print(f"Saved: {save_png}")
    return G, pos

def _to_numpy_image(
    img: Union[np.ndarray, "torch.Tensor"],
    mean: Tuple[float, float, float] = (0.5071, 0.4867, 0.4408),  # CIFAR-100 (common)
    std:  Tuple[float, float, float] = (0.2675, 0.2565, 0.2761),
) -> np.ndarray:
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().float()
        if img.ndim == 4:  # take first if batch
            img = img[0]
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = img.permute(1, 2, 0)  # CxHxW -> HxWxC
        img = img.numpy()
    else:
        img = np.asarray(img)

    if img.ndim == 2:
        img = img[..., None]

    if img.min() < -0.5 or img.max() > 1.5:
        mean_arr = np.array(mean).reshape(1,1,-1)
        std_arr  = np.array(std).reshape(1,1,-1)
        img = img * std_arr + mean_arr  # de-normalize to ~[0,1]

    img = np.clip(img, 0.0, 1.0).astype(np.float32)
    return img

if __name__ == "__main__":

    _, test_loader = get_loaders("./data", 1)
    
    recon_builder = CIFARReCoNBuilder(ReCoN(1))
    vision_model, device = load_model_from_pt('best_resnet18_cifar100_super.pt')

    for idx, (image, _ ) in enumerate(test_loader):
        image = image.to(device)
        
        previous_net_num = recon_builder.net.num
        recon_builder.build(image, vision_model)
        new_node_indices = range(previous_net_num, recon_builder.net.num)
        print(f"Processed image {idx}, added {len(new_node_indices)} nodes.")

        visualize_recon(recon_builder, title="ReCoN CIFAR 100", save_png=None, sample_image=image[0], new_node_indices=new_node_indices)
        plt.show()
        
 