import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def visualize_pcd_views(
        pcd_path: str,
        z_limits: tuple[float, float] = (-2.0, 2.0),
        point_size: float = 0.5,
        figsize: tuple[int, int] = (16, 8),  # ← increased default size
        cmap: str = "turbo"
    ) -> None:
    """
    Load a point-cloud file and show:
      1) Bird’s-eye view (X vs Y) with height colouring
      2) Side view       (X vs Z) with height colouring
    """
    # Load
    pcd = o3d.io.read_point_cloud(pcd_path)
    if len(pcd.points) == 0:
        raise ValueError(f"No points found in {pcd_path}")
    pts = np.asarray(pcd.points)

    # Height filter
    mask = (pts[:, 2] >= z_limits[0]) & (pts[:, 2] <= z_limits[1])
    pts = pts[mask]

    # Create subplots: bird’s-eye + side
    fig, (ax_bev, ax_side) = plt.subplots(1, 2, figsize=figsize)

    # Bird’s-eye view (X vs Y)
    sc1 = ax_bev.scatter(
        pts[:, 0], pts[:, 1], c=pts[:, 2],
        s=point_size, cmap=cmap, linewidth=0
    )
    ax_bev.set_aspect('equal', 'box')
    ax_bev.set_xlabel('X [m]')
    ax_bev.set_ylabel('Y [m]')
    ax_bev.set_title('Bird’s-Eye View')
    ax_bev.grid(True)
    cb1 = fig.colorbar(sc1, ax=ax_bev, fraction=0.046, pad=0.04)
    cb1.set_label('Height [m]')

    # Side view (X vs Z)
    sc2 = ax_side.scatter(
        pts[:, 0], pts[:, 2], c=pts[:, 2],
        s=point_size, cmap=cmap, linewidth=0
    )
    ax_side.set_aspect('equal', 'box')
    ax_side.set_xlabel('X [m]')
    ax_side.set_ylabel('Height Z [m]')
    ax_side.set_title('Side View (Elevation)')
    ax_side.grid(True)
    cb2 = fig.colorbar(sc2, ax=ax_side, fraction=0.046, pad=0.04)
    cb2.set_label('Height [m]')

    plt.tight_layout()
    plt.show()
