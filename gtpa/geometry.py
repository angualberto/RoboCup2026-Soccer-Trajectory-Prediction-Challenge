import numpy as np
import torch
from scipy.spatial import Delaunay, Voronoi

def compute_knn_adjacency(pos: torch.Tensor, k: int = 5) -> torch.Tensor:
    """
    Pure torch k-NN adjacency. No CPU copy, no scipy.
    pos: (batch, num_players, 2)
    returns: (batch, num_players, num_players) binary adjacency
    """
    B, N, _ = pos.shape
    diff = pos.unsqueeze(2) - pos.unsqueeze(1)
    dist = diff.norm(dim=-1)
    dist.diagonal(dim1=1, dim2=2).fill_(float('inf'))
    _, idx = dist.topk(k, dim=-1, largest=False)
    adj = torch.zeros(B, N, N, device=pos.device, dtype=torch.bool)
    adj.scatter_(2, idx, True)
    adj = adj | adj.transpose(1, 2)
    return adj.float()


def compute_delaunay_adjacency(positions, num_players=22):
    """
    Computes Delaunay Triangulation adjacency matrix for players.
    positions: numpy array of shape (num_players, 2)
    Returns:
        adjacency: binary numpy array of shape (num_players, num_players)
    """
    adj = np.zeros((num_players, num_players), dtype=np.float32)
    if len(positions) < 4:
        return adj
    try:
        tri = Delaunay(positions)
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    u, v = simplex[i], simplex[j]
                    if u < num_players and v < num_players:
                        adj[u, v] = 1.0
                        adj[v, u] = 1.0
    except Exception:
        # Fallback to fully connected or distance-based if Delaunay fails (e.g. collinear points)
        pass
    return adj

def get_bounded_voronoi_areas(positions, num_players=22, field_bounds=[-52.5, 52.5, -34.0, 34.0]):
    """
    Computes Voronoi cell areas for players, bounding the cells to the soccer field.
    positions: numpy array of shape (num_players, 2)
    Returns:
        areas: numpy array of shape (num_players,) representing the area of each player's Voronoi cell.
    """
    areas = np.zeros(num_players, dtype=np.float32)
    xmin, xmax, ymin, ymax = field_bounds
    
    # To handle boundary cells (which go to infinity), we add dummy boundary points around the field
    padding = 100.0
    dummy_pts = np.array([
        [xmin - padding, ymin - padding],
        [xmin - padding, ymax + padding],
        [xmax + padding, ymin - padding],
        [xmax + padding, ymax + padding],
        [xmin - padding, (ymin + ymax)/2],
        [xmax + padding, (ymin + ymax)/2],
        [(xmin + xmax)/2, ymin - padding],
        [(xmin + xmax)/2, ymax + padding],
    ])
    
    all_pts = np.vstack([positions[:num_players], dummy_pts])
    try:
        vor = Voronoi(all_pts)
        for i in range(num_players):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            if not region or -1 in region:
                # If unbounded even with dummies, assign a default small area
                areas[i] = 1.0
                continue
            
            # Compute polygon vertices
            vertices = vor.vertices[region]
            
            # Simple polygon area calculation (Shoelace formula)
            x = vertices[:, 0]
            y = vertices[:, 1]
            area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            areas[i] = min(area, (xmax - xmin) * (ymax - ymin))
    except Exception:
        areas.fill((xmax - xmin) * (ymax - ymin) / num_players)
        
    return areas

def compute_influence_tensor(positions, velocities, grid_resolution=(20, 20), field_bounds=[-52.5, 52.5, -34.0, 34.0]):
    """
    Computes the anisotropic spatial influence tensor for team L (first 11 players)
    and team R (next 11 players) over a grid.
    positions: numpy array of shape (22, 2)
    velocities: numpy array of shape (22, 2)
    Returns:
        influence_map: numpy array of shape grid_resolution
    """
    xmin, xmax, ymin, ymax = field_bounds
    x_grid = np.linspace(xmin, xmax, grid_resolution[0])
    y_grid = np.linspace(ymin, ymax, grid_resolution[1])
    X, Y = np.meshgrid(x_grid, y_grid)
    grid_pts = np.stack([X.ravel(), Y.ravel()], axis=-1) # (grid_res[0]*grid_res[1], 2)
    
    influence_L = np.zeros(grid_pts.shape[0], dtype=np.float32)
    influence_R = np.zeros(grid_pts.shape[0], dtype=np.float32)
    
    beta_f = 0.15
    beta_l = 2.0
    sigma_min = 2.0
    
    for i in range(22):
        pos = positions[i]
        vel = velocities[i]
        speed = np.linalg.norm(vel)
        
        # Calculate covariance matrix Sigma
        angle = np.arctan2(vel[1], vel[0]) if speed > 1e-3 else 0.0
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        
        sigma_f = beta_f * speed + sigma_min
        sigma_l = beta_l / (speed + 0.1)
        
        Sigma = R @ np.array([[sigma_f**2, 0], [0, sigma_l**2]]) @ R.T
        try:
            inv_Sigma = np.linalg.inv(Sigma)
        except np.linalg.LinAlgError:
            inv_Sigma = np.identity(2) / (sigma_min**2)
            
        diff = grid_pts - pos
        dist_term = np.sum(diff @ inv_Sigma * diff, axis=-1)
        inf = np.exp(-0.5 * dist_term)
        
        if i < 11:
            influence_L += inf
        else:
            influence_R += inf
            
    influence = (influence_L - influence_R) / (influence_L + influence_R + 1e-5)
    return influence.reshape(grid_resolution)

def to_polar_coordinates(ref_pos, target_pos):
    """
    Converts Cartesian positions relative to a reference position to polar coordinates (r, theta).
    """
    diff = target_pos - ref_pos
    r = np.linalg.norm(diff, axis=-1)
    theta = np.arctan2(diff[..., 1], diff[..., 0])
    return r, theta
