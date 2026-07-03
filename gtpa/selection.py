import numpy as np
import torch
from sklearn.cluster import DBSCAN

def select_best_trajectories(particles, num_particles=100, num_agents=23):
    """
    Selects the best trajectory from the particles using DBSCAN clustering on the final state.
    Args:
        particles: tensor of shape (horizon, 1, batch * P, 92)
        num_particles: P
        num_agents: 23 (l1..l11, r1..r11, ball)
    Returns:
        selected_trajectories: tensor of shape (horizon, 1, batch, 92)
    """
    device = particles.device
    horizon, _, total_particles, feat_dim = particles.size()
    batch_size = total_particles // num_particles
    P = num_particles
    
    # Selected output tensor of shape (horizon, 1, batch, 92)
    selected_trajectories = torch.zeros(horizon, 1, batch_size, feat_dim, device=device)
    
    # We perform clustering for each batch element independently
    particles_cpu = particles.detach().cpu().numpy() # (horizon, 1, batch * P, 92)
    
    for b in range(batch_size):
        # Extract states for this batch element
        # Shape: (P, horizon, 92)
        batch_particles = particles_cpu[:, 0, b * P : (b + 1) * P] # (horizon, P, 92)
        batch_particles = np.transpose(batch_particles, (1, 0, 2)) # (P, horizon, 92)
        
        # We cluster based on the final frame's evaluation targets: 11 left-team players + ball
        # Left-team is index 0 to 10 (11 players), Ball is index 22
        # Each agent has 4 features (x, y, vx, vy)
        # We select the final positions (x, y) at the last timestep
        final_states = batch_particles[:, -1] # (P, 92)
        
        # Extract evaluation feature coordinates: shape (P, 12 * 2) = (P, 24)
        eval_features = []
        for p in range(P):
            state = final_states[p].reshape(num_agents, 4)
            left_team_pos = state[:11, :2].flatten()
            ball_pos = state[22, :2].flatten()
            feat = np.concatenate([left_team_pos, ball_pos])
            eval_features.append(feat)
            
        eval_features = np.stack(eval_features, axis=0) # (P, 24)
        
        # Run DBSCAN
        # eps is the distance threshold (e.g. 5.0 meters on the field)
        # min_samples is the minimum size of a cluster
        eps = 6.0
        min_samples = max(2, int(P * 0.1))
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(eval_features)
        labels = db.labels_
        
        unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
        
        if len(unique_labels) > 0:
            # Largest cluster label
            best_label = unique_labels[np.argmax(counts)]
            cluster_indices = np.where(labels == best_label)[0]

            # Compute centroid of the largest cluster
            centroid = np.mean(eval_features[cluster_indices], axis=0)

            # Prefer particles that are both close to the final-state centroid and smooth over time.
            best_score = None
            best_particle_idx = cluster_indices[0]
            for particle_idx in cluster_indices:
                final_dist = np.linalg.norm(eval_features[particle_idx] - centroid, axis=-1)

                particle_traj = batch_particles[particle_idx]
                particle_states = particle_traj.reshape(horizon, num_agents, 4)
                positions = particle_states[:, :, :2]
                if positions.shape[0] >= 3:
                    accel = positions[2:] - 2.0 * positions[1:-1] + positions[:-2]
                    smoothness = np.mean(np.linalg.norm(accel, axis=-1))
                else:
                    smoothness = 0.0

                score = final_dist + 0.1 * smoothness
                if best_score is None or score < best_score:
                    best_score = score
                    best_particle_idx = particle_idx
        else:
            # Fallback: select the medoid (particle with minimum average distance to all others)
            dist_matrix = np.linalg.norm(eval_features[:, None, :] - eval_features[None, :, :], axis=-1)
            avg_dists = np.mean(dist_matrix, axis=1)
            best_particle_idx = np.argmin(avg_dists)
            
        # Write the selected trajectory back to the output tensor
        # Shape of selected trajectory: (horizon, 92)
        selected_trajectories[:, 0, b] = torch.tensor(batch_particles[best_particle_idx], device=device)
        
    return selected_trajectories
