# CLAUDE_ADDED
"""
Academic SVG Evaluator for Conference Paper
学会論文用SVG評価器
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_svg import FigureCanvasSVG
from typing import Dict, Any, List, Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score, r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
import seaborn as sns
from pathlib import Path


class AcademicSVGEvaluator:
    """Academic paper quality SVG output evaluator"""

    def __init__(self, output_dir: str, anonymize_subjects: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.anonymize_subjects = anonymize_subjects

        # Set Times New Roman font for academic papers
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'Liberation Serif']
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 12
        plt.rcParams['figure.titlesize'] = 18

        # Academic color palette
        self.colors = {
            'skill_high': '#2E86AB',    # Blue for high skill
            'skill_low': '#A23B72',     # Purple for low skill
            'skill_med': '#F18F01',     # Orange for medium skill
            'subjects': plt.cm.Set3,    # Colormap for subjects
            'grid': '#E5E5E5',          # Light gray for grid
            'text': '#2C2C2C'           # Dark gray for text
        }

    def anonymize_subject_names(self, subject_ids: List[str]) -> Dict[str, str]:
        """Convert subject names to anonymous labels (Subject1, Subject2, etc.)"""
        unique_subjects = sorted(list(set(subject_ids)))
        return {subj: f"Subject{i+1}" for i, subj in enumerate(unique_subjects)}

    def extract_latent_variables(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
                                device: torch.device) -> Dict[str, np.ndarray]:
        """Extract latent variables from model"""
        model.eval()
        all_z_style = []
        all_z_skill = []
        all_subject_ids = []
        all_skill_scores = []
        all_trajectories = []
        all_reconstructed = []

        with torch.no_grad():
            for batch in dataloader:
                trajectories = batch['trajectory'].to(device)
                subject_ids = batch['subject_id']
                skill_scores = batch['skill_score'].to(device)

                # Get model outputs
                outputs = model(trajectories, subject_ids, skill_scores)

                z_style = outputs['z_style'].cpu().numpy()
                z_skill = outputs['z_skill'].cpu().numpy()
                reconstructed = outputs['reconstructed'].cpu().numpy()

                all_z_style.append(z_style)
                all_z_skill.append(z_skill)
                all_subject_ids.extend(subject_ids)
                all_skill_scores.append(skill_scores.cpu().numpy())
                all_trajectories.append(trajectories.cpu().numpy())
                all_reconstructed.append(reconstructed)

        return {
            'z_style': np.concatenate(all_z_style, axis=0),
            'z_skill': np.concatenate(all_z_skill, axis=0),
            'subject_ids': all_subject_ids,
            'skill_scores': np.concatenate(all_skill_scores, axis=0),
            'trajectories': np.concatenate(all_trajectories, axis=0),
            'reconstructed': np.concatenate(all_reconstructed, axis=0)
        }

    def create_style_space_visualization(self, z_style: np.ndarray, subject_ids: List[str],
                                       save_path: str = None) -> plt.Figure:
        """Create style latent space visualization"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Anonymize subject names if needed
        if self.anonymize_subjects:
            subject_mapping = self.anonymize_subject_names(subject_ids)
            display_subjects = [subject_mapping[sid] for sid in subject_ids]
        else:
            display_subjects = subject_ids

        unique_subjects = sorted(list(set(display_subjects)))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_subjects)))

        # PCA visualization
        if z_style.shape[1] >= 2:
            pca = PCA(n_components=2)
            z_style_pca = pca.fit_transform(z_style)

            for i, subject in enumerate(unique_subjects):
                mask = np.array(display_subjects) == subject
                ax1.scatter(z_style_pca[mask, 0], z_style_pca[mask, 1],
                           c=[colors[i]], label=subject, alpha=0.7, s=50,
                           edgecolors='black', linewidth=0.5)

            ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
            ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
            ax1.set_title('Style Latent Space (PCA)')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)

        # t-SNE visualization
        if len(z_style) >= 30:  # Minimum samples for t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(z_style)//4))
            z_style_tsne = tsne.fit_transform(z_style)

            for i, subject in enumerate(unique_subjects):
                mask = np.array(display_subjects) == subject
                ax2.scatter(z_style_tsne[mask, 0], z_style_tsne[mask, 1],
                           c=[colors[i]], label=subject, alpha=0.7, s=50,
                           edgecolors='black', linewidth=0.5)

            ax2.set_xlabel('t-SNE Component 1')
            ax2.set_ylabel('t-SNE Component 2')
            ax2.set_title('Style Latent Space (t-SNE)')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, format='svg', bbox_inches='tight', dpi=300)

        return fig

    def create_skill_space_visualization(self, z_skill: np.ndarray, skill_scores: np.ndarray,
                                       subject_ids: List[str] = None, save_path: str = None) -> plt.Figure:
        """Create skill latent space visualization"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Normalize skill scores for color mapping
        skill_norm = (skill_scores - skill_scores.min()) / (skill_scores.max() - skill_scores.min() + 1e-8)

        # PCA visualization
        if z_skill.shape[1] >= 2:
            pca = PCA(n_components=2)
            z_skill_pca = pca.fit_transform(z_skill)

            scatter = ax1.scatter(z_skill_pca[:, 0], z_skill_pca[:, 1],
                                c=skill_scores, cmap='RdYlBu_r', alpha=0.7, s=50,
                                edgecolors='black', linewidth=0.5)

            ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
            ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
            ax1.set_title('Skill Latent Space (PCA)')

            cbar = plt.colorbar(scatter, ax=ax1)
            cbar.set_label('Skill Score')
            ax1.grid(True, alpha=0.3)

        # t-SNE visualization
        if len(z_skill) >= 30:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(z_skill)//4))
            z_skill_tsne = tsne.fit_transform(z_skill)

            scatter = ax2.scatter(z_skill_tsne[:, 0], z_skill_tsne[:, 1],
                                c=skill_scores, cmap='RdYlBu_r', alpha=0.7, s=50,
                                edgecolors='black', linewidth=0.5)

            ax2.set_xlabel('t-SNE Component 1')
            ax2.set_ylabel('t-SNE Component 2')
            ax2.set_title('Skill Latent Space (t-SNE)')

            cbar = plt.colorbar(scatter, ax=ax2)
            cbar.set_label('Skill Score')
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, format='svg', bbox_inches='tight', dpi=300)

        return fig

    def create_skill_regression_analysis(self, z_skill: np.ndarray, skill_scores: np.ndarray,
                                       save_path: str = None) -> Tuple[plt.Figure, Dict[str, float]]:
        """Create skill regression performance analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        results = {}

        # Linear Regression
        lr = LinearRegression()
        lr.fit(z_skill, skill_scores)
        lr_pred = lr.predict(z_skill)
        lr_r2 = r2_score(skill_scores, lr_pred)
        lr_rmse = np.sqrt(mean_squared_error(skill_scores, lr_pred))

        ax1.scatter(skill_scores, lr_pred, alpha=0.6, color=self.colors['skill_high'])
        ax1.plot([skill_scores.min(), skill_scores.max()],
                [skill_scores.min(), skill_scores.max()], 'r--', lw=2)
        ax1.set_xlabel('True Skill Score')
        ax1.set_ylabel('Predicted Skill Score')
        ax1.set_title(f'Linear Regression\\nR² = {lr_r2:.4f}, RMSE = {lr_rmse:.4f}')
        ax1.grid(True, alpha=0.3)

        results['linear_r2'] = lr_r2
        results['linear_rmse'] = lr_rmse

        # SVM Regression
        svr = SVR(kernel='rbf', C=1.0, gamma='scale')
        svr.fit(z_skill, skill_scores)
        svr_pred = svr.predict(z_skill)
        svr_r2 = r2_score(skill_scores, svr_pred)
        svr_rmse = np.sqrt(mean_squared_error(skill_scores, svr_pred))

        ax2.scatter(skill_scores, svr_pred, alpha=0.6, color=self.colors['skill_med'])
        ax2.plot([skill_scores.min(), skill_scores.max()],
                [skill_scores.min(), skill_scores.max()], 'r--', lw=2)
        ax2.set_xlabel('True Skill Score')
        ax2.set_ylabel('Predicted Skill Score')
        ax2.set_title(f'SVM Regression\\nR² = {svr_r2:.4f}, RMSE = {svr_rmse:.4f}')
        ax2.grid(True, alpha=0.3)

        results['svm_r2'] = svr_r2
        results['svm_rmse'] = svr_rmse

        # MLP Regression
        mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
        mlp.fit(z_skill, skill_scores)
        mlp_pred = mlp.predict(z_skill)
        mlp_r2 = r2_score(skill_scores, mlp_pred)
        mlp_rmse = np.sqrt(mean_squared_error(skill_scores, mlp_pred))

        ax3.scatter(skill_scores, mlp_pred, alpha=0.6, color=self.colors['skill_low'])
        ax3.plot([skill_scores.min(), skill_scores.max()],
                [skill_scores.min(), skill_scores.max()], 'r--', lw=2)
        ax3.set_xlabel('True Skill Score')
        ax3.set_ylabel('Predicted Skill Score')
        ax3.set_title(f'MLP Regression\\nR² = {mlp_r2:.4f}, RMSE = {mlp_rmse:.4f}')
        ax3.grid(True, alpha=0.3)

        results['mlp_r2'] = mlp_r2
        results['mlp_rmse'] = mlp_rmse

        # Cross-validation comparison
        methods = ['Linear', 'SVM', 'MLP']
        r2_scores = [lr_r2, svr_r2, mlp_r2]
        rmse_scores = [lr_rmse, svr_rmse, mlp_rmse]

        x_pos = np.arange(len(methods))
        ax4.bar(x_pos - 0.2, r2_scores, 0.4, label='R² Score', color=self.colors['skill_high'])
        ax4_twin = ax4.twinx()
        ax4_twin.bar(x_pos + 0.2, rmse_scores, 0.4, label='RMSE', color=self.colors['skill_low'])

        ax4.set_xlabel('Regression Method')
        ax4.set_ylabel('R² Score', color=self.colors['skill_high'])
        ax4_twin.set_ylabel('RMSE', color=self.colors['skill_low'])
        ax4.set_title('Regression Performance Comparison')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(methods)
        ax4.grid(True, alpha=0.3)

        # Add best performance indicator
        best_method = methods[np.argmax(r2_scores)]
        results['best_method'] = best_method
        results['best_r2'] = max(r2_scores)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, format='svg', bbox_inches='tight', dpi=300)

        return fig, results

    def create_reconstruction_analysis(self, trajectories: np.ndarray, reconstructed: np.ndarray,
                                     subject_ids: List[str] = None, save_path: str = None) -> plt.Figure:
        """Create trajectory reconstruction analysis"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Calculate reconstruction errors
        mse_pos_x = np.mean((trajectories[:, :, 0] - reconstructed[:, :, 0])**2, axis=1)
        mse_pos_y = np.mean((trajectories[:, :, 1] - reconstructed[:, :, 1])**2, axis=1)
        mse_vel_x = np.mean((trajectories[:, :, 2] - reconstructed[:, :, 2])**2, axis=1)
        mse_vel_y = np.mean((trajectories[:, :, 3] - reconstructed[:, :, 3])**2, axis=1)
        mse_acc_x = np.mean((trajectories[:, :, 4] - reconstructed[:, :, 4])**2, axis=1)
        mse_acc_y = np.mean((trajectories[:, :, 5] - reconstructed[:, :, 5])**2, axis=1)

        # Plot histograms for each component
        components = ['Position X', 'Position Y', 'Velocity X', 'Velocity Y', 'Acceleration X', 'Acceleration Y']
        mse_values = [mse_pos_x, mse_pos_y, mse_vel_x, mse_vel_y, mse_acc_x, mse_acc_y]

        for i, (component, mse) in enumerate(zip(components, mse_values)):
            ax = axes[i//3, i%3]
            ax.hist(mse, bins=30, alpha=0.7, color=self.colors['skill_high'], edgecolor='black')
            ax.set_xlabel('MSE')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{component}\\nMean MSE: {np.mean(mse):.6f}')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, format='svg', bbox_inches='tight', dpi=300)

        return fig

    def create_trajectory_samples(self, trajectories: np.ndarray, reconstructed: np.ndarray,
                                subject_ids: List[str] = None, n_samples: int = 6,
                                save_path: str = None) -> plt.Figure:
        """Create trajectory reconstruction samples"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Anonymize subject names if needed
        if self.anonymize_subjects and subject_ids:
            subject_mapping = self.anonymize_subject_names(subject_ids)
            display_subjects = [subject_mapping[sid] for sid in subject_ids]
        else:
            display_subjects = subject_ids if subject_ids else [f"Sample{i+1}" for i in range(len(trajectories))]

        # Select random samples
        indices = np.random.choice(len(trajectories), min(n_samples, len(trajectories)), replace=False)

        for i, idx in enumerate(indices):
            ax = axes[i//3, i%3]

            # Plot original trajectory
            ax.plot(trajectories[idx, :, 0], trajectories[idx, :, 1],
                   'b-', linewidth=2, label='Original', alpha=0.8)

            # Plot reconstructed trajectory
            ax.plot(reconstructed[idx, :, 0], reconstructed[idx, :, 1],
                   'r--', linewidth=2, label='Reconstructed', alpha=0.8)

            # Mark start and end points
            ax.scatter(trajectories[idx, 0, 0], trajectories[idx, 0, 1],
                      c='green', s=100, marker='o', label='Start', zorder=5)
            ax.scatter(trajectories[idx, -1, 0], trajectories[idx, -1, 1],
                      c='red', s=100, marker='s', label='End', zorder=5)

            ax.set_xlabel('Position X')
            ax.set_ylabel('Position Y')
            if subject_ids:
                ax.set_title(f'{display_subjects[idx]}')
            else:
                ax.set_title(f'Sample {idx+1}')

            if i == 0:
                ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, format='svg', bbox_inches='tight', dpi=300)

        return fig

    def evaluate_model(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
                      device: torch.device, experiment_name: str = "experiment") -> Dict[str, Any]:
        """Complete model evaluation with SVG outputs"""
        print(f"Starting academic evaluation for {experiment_name}")

        # Extract data
        data = self.extract_latent_variables(model, dataloader, device)

        results = {}

        # 1. Style space analysis
        print("Creating style space visualization...")
        style_fig = self.create_style_space_visualization(
            data['z_style'], data['subject_ids'],
            save_path=self.output_dir / f"{experiment_name}_style_space.svg"
        )
        results['style_space_fig'] = style_fig

        # 2. Skill space analysis
        print("Creating skill space visualization...")
        skill_fig = self.create_skill_space_visualization(
            data['z_skill'], data['skill_scores'], data['subject_ids'],
            save_path=self.output_dir / f"{experiment_name}_skill_space.svg"
        )
        results['skill_space_fig'] = skill_fig

        # 3. Skill regression analysis
        print("Creating skill regression analysis...")
        regression_fig, regression_results = self.create_skill_regression_analysis(
            data['z_skill'], data['skill_scores'],
            save_path=self.output_dir / f"{experiment_name}_skill_regression.svg"
        )
        results['skill_regression_fig'] = regression_fig
        results['skill_regression_results'] = regression_results

        # 4. Reconstruction analysis
        print("Creating reconstruction analysis...")
        reconstruction_fig = self.create_reconstruction_analysis(
            data['trajectories'], data['reconstructed'], data['subject_ids'],
            save_path=self.output_dir / f"{experiment_name}_reconstruction.svg"
        )
        results['reconstruction_fig'] = reconstruction_fig

        # 5. Trajectory samples
        print("Creating trajectory samples...")
        trajectory_fig = self.create_trajectory_samples(
            data['trajectories'], data['reconstructed'], data['subject_ids'],
            save_path=self.output_dir / f"{experiment_name}_trajectory_samples.svg"
        )
        results['trajectory_samples_fig'] = trajectory_fig

        # Calculate summary metrics
        results['summary_metrics'] = {
            'reconstruction_mse': np.mean((data['trajectories'] - data['reconstructed'])**2),
            'skill_regression_r2': regression_results['best_r2'],
            'best_regression_method': regression_results['best_method'],
            'n_subjects': len(set(data['subject_ids'])),
            'n_samples': len(data['trajectories'])
        }

        print(f"Evaluation complete. Results saved to {self.output_dir}")
        return results