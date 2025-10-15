# CLAUDE_ADDED
"""
Academic Comprehensive Evaluator for Conference Papers
学会論文用包括評価器
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score, r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from pathlib import Path
import seaborn as sns
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class AcademicComprehensiveEvaluator:
    """Academic paper quality comprehensive evaluator using existing evaluator logic"""

    def __init__(self, output_dir: str, anonymize_subjects: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.anonymize_subjects = anonymize_subjects

        # Set Times New Roman font for academic papers (optimized for 3.5cm width)
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'Liberation Serif']
        plt.rcParams['font.size'] = 8
        plt.rcParams['axes.labelsize'] = 9
        plt.rcParams['axes.titlesize'] = 10
        plt.rcParams['xtick.labelsize'] = 7
        plt.rcParams['ytick.labelsize'] = 7
        plt.rcParams['legend.fontsize'] = 7
        plt.rcParams['figure.titlesize'] = 11
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'

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

    def save_figure(self, fig: plt.Figure, save_path: str = None):
        """Save figure in both PDF and high-quality PNG formats"""
        if save_path:
            pdf_path = Path(save_path)
            png_path = pdf_path.with_suffix('.png')

            # Save PDF (vector format, scalable, preferred for academic papers)
            fig.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300,
                       facecolor='white', edgecolor='none')

            # Save PNG (raster format, 300 DPI for print quality)
            fig.savefig(png_path, format='png', bbox_inches='tight', dpi=300,
                       facecolor='white', edgecolor='none')

            print(f"Saved: {pdf_path.name} and {png_path.name}")

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
                # Handle both tuple and dict formats
                if isinstance(batch, (list, tuple)):
                    # Tuple format: (trajectories, subject_ids, skill_scores)
                    trajectories, subject_ids, skill_scores = batch
                    trajectories = trajectories.to(device)
                    skill_scores = skill_scores.to(device)
                else:
                    # Dict format
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
                                       save_path: str = None) -> Tuple[plt.Figure, Dict[str, Any]]:
        """Create style classification analysis using StyleClassificationEvaluator logic"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.svm import SVC
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(6.7, 5.0))

        # Anonymize subject names if needed
        if self.anonymize_subjects:
            subject_mapping = self.anonymize_subject_names(subject_ids)
            display_subjects = [subject_mapping[sid] for sid in subject_ids]
        else:
            display_subjects = subject_ids

        results = {}

        # Data preprocessing
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(display_subjects)
        label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(z_style)

        # Remove NaN values
        valid_indices = ~(np.isnan(X_scaled).any(axis=1) | np.isnan(y_encoded))
        X_clean = X_scaled[valid_indices]
        y_clean = y_encoded[valid_indices]

        if len(np.unique(y_clean)) < 2:
            for ax in [ax1, ax2, ax3, ax4]:
                ax.text(0.5, 0.5, 'Insufficient subjects for classification',
                       transform=ax.transAxes, ha='center')
            plt.tight_layout()
            self.save_figure(fig, save_path)
            return fig, {'error': 'insufficient_subjects'}

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=0.2, random_state=42,
            stratify=y_clean if len(np.unique(y_clean)) > 1 else None
        )

        # 1. MLP Classification
        mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        mlp.fit(X_train, y_train)
        mlp_pred = mlp.predict(X_test)
        mlp_acc = accuracy_score(y_test, mlp_pred)
        mlp_cm = confusion_matrix(y_test, mlp_pred)

        im1 = ax1.imshow(mlp_cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax1.set_title(f'MLP Confusion Matrix\nAccuracy: {mlp_acc:.4f}')
        tick_marks = np.arange(len(label_mapping))
        ax1.set_xticks(tick_marks)
        ax1.set_yticks(tick_marks)
        ax1.set_xticklabels([label_mapping[i] for i in range(len(label_mapping))], rotation=45)
        ax1.set_yticklabels([label_mapping[i] for i in range(len(label_mapping))])
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')

        # Add text annotations to confusion matrix
        thresh = mlp_cm.max() / 2.
        for i, j in np.ndindex(mlp_cm.shape):
            ax1.text(j, i, format(mlp_cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if mlp_cm[i, j] > thresh else "black")

        results['mlp_accuracy'] = mlp_acc
        results['mlp_confusion_matrix'] = mlp_cm

        # 2. SVM Classification
        svm = SVC(kernel='rbf', random_state=42)
        svm.fit(X_train, y_train)
        svm_pred = svm.predict(X_test)
        svm_acc = accuracy_score(y_test, svm_pred)
        svm_cm = confusion_matrix(y_test, svm_pred)

        im2 = ax2.imshow(svm_cm, interpolation='nearest', cmap=plt.cm.Greens)
        ax2.set_title(f'SVM Confusion Matrix\nAccuracy: {svm_acc:.4f}')
        ax2.set_xticks(tick_marks)
        ax2.set_yticks(tick_marks)
        ax2.set_xticklabels([label_mapping[i] for i in range(len(label_mapping))], rotation=45)
        ax2.set_yticklabels([label_mapping[i] for i in range(len(label_mapping))])
        ax2.set_ylabel('True Label')
        ax2.set_xlabel('Predicted Label')

        # Add text annotations
        thresh = svm_cm.max() / 2.
        for i, j in np.ndindex(svm_cm.shape):
            ax2.text(j, i, format(svm_cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if svm_cm[i, j] > thresh else "black")

        results['svm_accuracy'] = svm_acc
        results['svm_confusion_matrix'] = svm_cm

        # 3. Random Forest Classification
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_acc = accuracy_score(y_test, rf_pred)
        rf_cm = confusion_matrix(y_test, rf_pred)

        im3 = ax3.imshow(rf_cm, interpolation='nearest', cmap=plt.cm.Oranges)
        ax3.set_title(f'Random Forest Confusion Matrix\nAccuracy: {rf_acc:.4f}')
        ax3.set_xticks(tick_marks)
        ax3.set_yticks(tick_marks)
        ax3.set_xticklabels([label_mapping[i] for i in range(len(label_mapping))], rotation=45)
        ax3.set_yticklabels([label_mapping[i] for i in range(len(label_mapping))])
        ax3.set_ylabel('True Label')
        ax3.set_xlabel('Predicted Label')

        # Add text annotations
        thresh = rf_cm.max() / 2.
        for i, j in np.ndindex(rf_cm.shape):
            ax3.text(j, i, format(rf_cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if rf_cm[i, j] > thresh else "black")

        results['rf_accuracy'] = rf_acc
        results['rf_confusion_matrix'] = rf_cm

        # 4. Accuracy Comparison
        methods = ['MLP', 'SVM', 'RandomForest']
        accuracies = [mlp_acc, svm_acc, rf_acc]
        colors_bars = [self.colors['skill_high'], self.colors['skill_med'], self.colors['skill_low']]

        bars = ax4.bar(methods, accuracies, color=colors_bars)
        ax4.set_ylabel('Classification Accuracy')
        ax4.set_title('Subject Classification Performance')
        ax4.set_ylim(0, 1)
        ax4.grid(True, alpha=0.3)

        # Add accuracy values on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')

        # Find best method
        best_idx = np.argmax(accuracies)
        results['best_method'] = methods[best_idx]
        results['best_accuracy'] = accuracies[best_idx]

        # Summary statistics
        results['n_subjects'] = len(label_mapping)
        results['n_samples'] = len(y_clean)
        results['train_samples'] = len(y_train)
        results['test_samples'] = len(y_test)

        plt.tight_layout()
        self.save_figure(fig, save_path)
        return fig, results

    def create_skill_space_visualization(self, z_skill: np.ndarray, skill_scores: np.ndarray,
                                       subject_ids: List[str] = None, save_path: str = None) -> plt.Figure:
        """Create skill latent space visualization using PCA only"""
        fig, ax = plt.subplots(1, 1, figsize=(3.35, 2.8))

        # PCA visualization
        if z_skill.shape[1] >= 2:
            pca = PCA(n_components=2)
            z_skill_pca = pca.fit_transform(z_skill)

            scatter = ax.scatter(z_skill_pca[:, 0], z_skill_pca[:, 1],
                                c=skill_scores, cmap='RdYlBu_r', alpha=0.7, s=10,
                                edgecolors='black', linewidth=0.5)

            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
            # ax.set_title(f'Skill Latent Space (PCA)\nExplained: {pca.explained_variance_ratio_.sum():.3f}')

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Skill Score')
            ax.grid(True, alpha=0.3)

            # Add statistics
            if subject_ids:
                stats_text = f'Samples: {len(z_skill)}\nSubjects: {len(set(subject_ids))}\nSkill Range: [{np.min(skill_scores):.3f}, {np.max(skill_scores):.3f}]'
            else:
                stats_text = f'Samples: {len(z_skill)}\nSkill Range: [{np.min(skill_scores):.3f}, {np.max(skill_scores):.3f}]'

            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=6)

        plt.tight_layout()
        self.save_figure(fig, save_path)
        return fig

    def create_skill_regression_analysis(self, z_skill: np.ndarray, skill_scores: np.ndarray,
                                       save_path: str = None) -> Tuple[plt.Figure, Dict[str, float]]:
        """Create skill regression performance analysis using SkillScoreRegressionEvaluator logic"""
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(6.7, 5.0))

        results = {}

        # Data preprocessing
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(z_skill)

        # Remove NaN values
        valid_indices = ~(np.isnan(X_scaled).any(axis=1) | np.isnan(skill_scores))
        X_clean = X_scaled[valid_indices]
        y_clean = np.array(skill_scores)[valid_indices]

        if len(y_clean) < 20:
            for ax in [ax1, ax2, ax3, ax4]:
                ax.text(0.5, 0.5, 'Insufficient samples for regression',
                       transform=ax.transAxes, ha='center')
            plt.tight_layout()
            self.save_figure(fig, save_path)
            return fig, {'error': 'insufficient_samples'}

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=0.2, random_state=42
        )

        # 1. Linear Regression
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        lr_pred = lr.predict(X_test)
        lr_r2 = r2_score(y_test, lr_pred)
        lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))

        ax1.scatter(y_test, lr_pred, alpha=0.6, color=self.colors['skill_high'])
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax1.set_xlabel('True Skill Score')
        ax1.set_ylabel('Predicted Skill Score')
        ax1.set_title(f'Linear Regression\nR² = {lr_r2:.4f}, RMSE = {lr_rmse:.4f}')
        ax1.grid(True, alpha=0.3)

        results['linear_r2'] = lr_r2
        results['linear_rmse'] = lr_rmse

        # 2. SVM Regression
        svr = SVR(kernel='rbf', C=1.0, gamma='scale')
        svr.fit(X_train, y_train)
        svr_pred = svr.predict(X_test)
        svr_r2 = r2_score(y_test, svr_pred)
        svr_rmse = np.sqrt(mean_squared_error(y_test, svr_pred))

        ax2.scatter(y_test, svr_pred, alpha=0.6, color=self.colors['skill_med'])
        ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax2.set_xlabel('True Skill Score')
        ax2.set_ylabel('Predicted Skill Score')
        ax2.set_title(f'SVM Regression\nR² = {svr_r2:.4f}, RMSE = {svr_rmse:.4f}')
        ax2.grid(True, alpha=0.3)

        results['svm_r2'] = svr_r2
        results['svm_rmse'] = svr_rmse

        # 3. MLP Regression - try multiple configurations like original evaluator
        mlp_configs = [
            {'hidden_layer_sizes': (50,), 'name': 'MLP-50'},
            {'hidden_layer_sizes': (100,), 'name': 'MLP-100'},
            {'hidden_layer_sizes': (50, 25), 'name': 'MLP-50-25'},
            {'hidden_layer_sizes': (100, 50), 'name': 'MLP-100-50'},
        ]

        best_mlp_r2 = -float('inf')
        best_mlp_pred = None

        for config in mlp_configs:
            try:
                mlp = MLPRegressor(
                    hidden_layer_sizes=config['hidden_layer_sizes'],
                    activation='relu',
                    solver='adam',
                    alpha=0.001,
                    max_iter=1000,
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.1
                )
                mlp.fit(X_train, y_train)
                pred = mlp.predict(X_test)
                r2 = r2_score(y_test, pred)
                if r2 > best_mlp_r2:
                    best_mlp_r2 = r2
                    best_mlp_pred = pred
            except:
                continue

        mlp_r2 = best_mlp_r2
        mlp_pred = best_mlp_pred if best_mlp_pred is not None else np.zeros_like(y_test)
        mlp_rmse = np.sqrt(mean_squared_error(y_test, mlp_pred))

        ax3.scatter(y_test, mlp_pred, alpha=0.6, color=self.colors['skill_low'])
        ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax3.set_xlabel('True Skill Score')
        ax3.set_ylabel('Predicted Skill Score')
        ax3.set_title(f'MLP Regression\nR² = {mlp_r2:.4f}, RMSE = {mlp_rmse:.4f}')
        ax3.grid(True, alpha=0.3)

        results['mlp_r2'] = mlp_r2
        results['mlp_rmse'] = mlp_rmse

        # 4. Comparison plot
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
        self.save_figure(fig, save_path)
        return fig, results

    def _calculate_r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R² score for reconstruction accuracy"""
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()

        # Remove any NaN values
        valid_mask = ~(np.isnan(y_true_flat) | np.isnan(y_pred_flat))
        y_true_clean = y_true_flat[valid_mask]
        y_pred_clean = y_pred_flat[valid_mask]

        if len(y_true_clean) == 0:
            return 0.0

        ss_res = np.sum((y_true_clean - y_pred_clean) ** 2)
        ss_tot = np.sum((y_true_clean - np.mean(y_true_clean)) ** 2)

        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0

        return 1 - (ss_res / ss_tot)

    def _calculate_and_print_rmse(self, trajectories: np.ndarray, reconstructed: np.ndarray):
        """Calculate and print RMSE for position, velocity, and acceleration components"""
        print("\n" + "="*60)
        print("TRAJECTORY RECONSTRUCTION RMSE ANALYSIS")
        print("="*60)

        # Position RMSE (X, Y components: indices 0, 1)
        pos_x_rmse = np.sqrt(np.mean((trajectories[:, :, 0] - reconstructed[:, :, 0])**2))
        pos_y_rmse = np.sqrt(np.mean((trajectories[:, :, 1] - reconstructed[:, :, 1])**2))
        pos_overall_rmse = np.sqrt(np.mean((trajectories[:, :, 0:2] - reconstructed[:, :, 0:2])**2))

        print(f"Position RMSE:")
        print(f"  X-component: {pos_x_rmse:.6f} [m]")
        print(f"  Y-component: {pos_y_rmse:.6f} [m]")
        print(f"  Overall:     {pos_overall_rmse:.6f} [m]")

        # Velocity RMSE (X, Y components: indices 2, 3)
        vel_x_rmse = np.sqrt(np.mean((trajectories[:, :, 2] - reconstructed[:, :, 2])**2))
        vel_y_rmse = np.sqrt(np.mean((trajectories[:, :, 3] - reconstructed[:, :, 3])**2))
        vel_overall_rmse = np.sqrt(np.mean((trajectories[:, :, 2:4] - reconstructed[:, :, 2:4])**2))

        print(f"\nVelocity RMSE:")
        print(f"  X-component: {vel_x_rmse:.6f} [m/s]")
        print(f"  Y-component: {vel_y_rmse:.6f} [m/s]")
        print(f"  Overall:     {vel_overall_rmse:.6f} [m/s]")

        # Acceleration RMSE (X, Y components: indices 4, 5)
        acc_x_rmse = np.sqrt(np.mean((trajectories[:, :, 4] - reconstructed[:, :, 4])**2))
        acc_y_rmse = np.sqrt(np.mean((trajectories[:, :, 5] - reconstructed[:, :, 5])**2))
        acc_overall_rmse = np.sqrt(np.mean((trajectories[:, :, 4:6] - reconstructed[:, :, 4:6])**2))

        print(f"\nAcceleration RMSE:")
        print(f"  X-component: {acc_x_rmse:.6f} [m/s²]")
        print(f"  Y-component: {acc_y_rmse:.6f} [m/s²]")
        print(f"  Overall:     {acc_overall_rmse:.6f} [m/s²]")

        # Total trajectory RMSE
        total_rmse = np.sqrt(np.mean((trajectories - reconstructed)**2))
        print(f"\nTotal Trajectory RMSE: {total_rmse:.6f}")

        # Calculate reconstruction accuracy (R² scores)
        print("\nRECONSTRUCTION ACCURACY (R² Score):")
        print("-" * 40)

        # Position accuracy
        pos_r2_x = self._calculate_r2(trajectories[:, :, 0], reconstructed[:, :, 0])
        pos_r2_y = self._calculate_r2(trajectories[:, :, 1], reconstructed[:, :, 1])
        pos_r2_overall = self._calculate_r2(trajectories[:, :, 0:2].flatten(), reconstructed[:, :, 0:2].flatten())

        print(f"Position R²:")
        print(f"  X-component: {pos_r2_x:.6f}")
        print(f"  Y-component: {pos_r2_y:.6f}")
        print(f"  Overall:     {pos_r2_overall:.6f}")

        # Velocity accuracy
        vel_r2_x = self._calculate_r2(trajectories[:, :, 2], reconstructed[:, :, 2])
        vel_r2_y = self._calculate_r2(trajectories[:, :, 3], reconstructed[:, :, 3])
        vel_r2_overall = self._calculate_r2(trajectories[:, :, 2:4].flatten(), reconstructed[:, :, 2:4].flatten())

        print(f"\nVelocity R²:")
        print(f"  X-component: {vel_r2_x:.6f}")
        print(f"  Y-component: {vel_r2_y:.6f}")
        print(f"  Overall:     {vel_r2_overall:.6f}")

        # Acceleration accuracy
        acc_r2_x = self._calculate_r2(trajectories[:, :, 4], reconstructed[:, :, 4])
        acc_r2_y = self._calculate_r2(trajectories[:, :, 5], reconstructed[:, :, 5])
        acc_r2_overall = self._calculate_r2(trajectories[:, :, 4:6].flatten(), reconstructed[:, :, 4:6].flatten())

        print(f"\nAcceleration R²:")
        print(f"  X-component: {acc_r2_x:.6f}")
        print(f"  Y-component: {acc_r2_y:.6f}")
        print(f"  Overall:     {acc_r2_overall:.6f}")

        # Total trajectory accuracy
        total_r2 = self._calculate_r2(trajectories.flatten(), reconstructed.flatten())
        print(f"\nTotal Trajectory R²: {total_r2:.6f}")
        print("="*60 + "\n")

    def create_trajectory_overlay_analysis(self, trajectories: np.ndarray, reconstructed: np.ndarray,
                                         subject_ids: List[str] = None, n_samples: int = 6,
                                         save_path: str = None) -> plt.Figure:
        """Create trajectory overlay analysis using TrajectoryOverlayEvaluator logic"""
        from matplotlib.gridspec import GridSpec

        # Calculate RMSE for position, velocity, and acceleration components
        self._calculate_and_print_rmse(trajectories, reconstructed)

        # Components to plot: 2D position, position time series, velocity time series, acceleration time series
        components = ['position_2d', 'position_time_series', 'velocity', 'acceleration']
        n_components = len(components)

        # Select samples randomly
        np.random.seed(42)  # For reproducibility
        n_samples = min(n_samples, len(trajectories))
        indices = np.random.choice(len(trajectories), n_samples, replace=False)

        # Create figure with GridSpec layout
        fig = plt.figure(figsize=(3.35, 10.0))
        gs = GridSpec(n_components, 1, figure=fig, hspace=0.3)

        # Anonymize subject names if needed
        if self.anonymize_subjects and subject_ids:
            subject_mapping = self.anonymize_subject_names(subject_ids)
            display_subjects = [subject_mapping[sid] for sid in subject_ids]
        else:
            display_subjects = subject_ids if subject_ids else [f"Sample{i+1}" for i in range(len(trajectories))]

        # Color scheme for different samples
        distinct_colors = [
            '#FF0000',  # Red
            '#0000FF',  # Blue
            '#008000',  # Green
            '#FF8000',  # Orange
            '#800080',  # Purple
            '#FF1493',  # Pink
            '#00CED1',  # Turquoise
            '#8B4513',  # Brown
            '#FFD700',  # Gold
            '#DC143C',  # Crimson
            '#4B0082',  # Indigo
            '#32CD32'   # Lime Green
        ]

        for comp_idx, component in enumerate(components):
            ax = fig.add_subplot(gs[comp_idx, 0])

            # Determine component indices and plot type
            if component == 'position_2d':
                comp_indices = [0, 1]  # x, y position
                plot_type = '2d'
                component_name = 'Position 2D'
                comp_labels = ['X Position [m]', 'Y Position [m]']
            elif component == 'position_time_series':
                comp_indices = [0, 1]  # x, y position
                plot_type = 'time_series'
                component_name = 'Position'
                comp_labels = ['X Position [m]', 'Y Position [m]']
                y_label = 'Position [m]'
            elif component == 'velocity':
                comp_indices = [2, 3]  # x, y velocity
                plot_type = 'time_series'
                component_name = 'Velocity'
                comp_labels = ['X Velocity [m/s]', 'Y Velocity [m/s]']
                y_label = 'Velocity [m/s]'
            elif component == 'acceleration':
                comp_indices = [4, 5]  # x, y acceleration
                plot_type = 'time_series'
                component_name = 'Acceleration'
                comp_labels = ['X Acceleration [m/s²]', 'Y Acceleration [m/s²]']
                y_label = 'Acceleration [m/s²]'

            if plot_type == 'time_series':
                # Time series plot (for velocity & acceleration)
                time_steps = np.arange(trajectories.shape[1])

                for i, idx in enumerate(indices):
                    color = distinct_colors[i % len(distinct_colors)]
                    subject_label = display_subjects[idx] if subject_ids else f"Sample{i+1}"

                    # X component
                    ax.plot(time_steps, trajectories[idx, :, comp_indices[0]],
                           color=color, alpha=0.7, linewidth=1,
                           label=f'Original {subject_label} (X)',
                           linestyle='-')
                    ax.plot(time_steps, reconstructed[idx, :, comp_indices[0]],
                           color=color, alpha=0.7, linewidth=1,
                           label=f'Reconstructed {subject_label} (X)',
                           linestyle='--')

                    # Y component (different color tone)
                    darker_color = distinct_colors[(i + 6) % len(distinct_colors)]
                    ax.plot(time_steps, trajectories[idx, :, comp_indices[1]],
                           color=darker_color, alpha=0.7, linewidth=1,
                           label=f'Original {subject_label} (Y)',
                           linestyle='-')
                    ax.plot(time_steps, reconstructed[idx, :, comp_indices[1]],
                           color=darker_color, alpha=0.7, linewidth=1,
                           label=f'Reconstructed {subject_label} (Y)',
                           linestyle='--')

                # Time series plot settings
                ax.set_xlabel('Time Steps [-]')
                ax.set_ylabel(y_label)
                # ax.set_title(f'{component_name} Time Series')
                ax.grid(True, alpha=0.3)

            else:
                # 2D trajectory plot (for position)
                for i, idx in enumerate(indices):
                    color = distinct_colors[i % len(distinct_colors)]
                    subject_label = display_subjects[idx] if subject_ids else f"Sample{i+1}"

                    # Original trajectory (solid line)
                    ax.plot(trajectories[idx, :, comp_indices[0]],
                           trajectories[idx, :, comp_indices[1]],
                           color=color, alpha=0.7, linewidth=1,
                           label=f'Original {subject_label}',
                           linestyle='-')

                    # Reconstructed trajectory (dashed line)
                    ax.plot(reconstructed[idx, :, comp_indices[0]],
                           reconstructed[idx, :, comp_indices[1]],
                           color=color, alpha=0.7, linewidth=1,
                           label=f'Reconstructed {subject_label}',
                           linestyle='--')

                    # Mark start and end points
                    ax.scatter(trajectories[idx, 0, comp_indices[0]],
                              trajectories[idx, 0, comp_indices[1]],
                              color=color, s=30, marker='o', edgecolor='black', linewidth=1,
                              zorder=10)
                    ax.scatter(trajectories[idx, -1, comp_indices[0]],
                              trajectories[idx, -1, comp_indices[1]],
                              color=color, s=30, marker='s', edgecolor='black', linewidth=1,
                              zorder=10)

                # 2D plot settings
                ax.set_xlabel(comp_labels[0])
                ax.set_ylabel(comp_labels[1])
                # ax.set_title(f'{component_name}')
                ax.grid(True, alpha=0.3)
                ax.axis('equal')

            # Legend (only for first subplot)
            if comp_idx == 0:
                # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
                pass

        # Overall title
        # fig.suptitle('Trajectory Overlay Comparison: Original (solid) vs Reconstructed (dashed)',
        #              fontsize=16, y=0.95)

        # Legend explanation
        # fig.text(0.02, 0.02, 'Circle: Start point, Square: End point', fontsize=10, style='italic')

        plt.tight_layout()
        self.save_figure(fig, save_path)
        return fig

    def create_orthogonality_analysis(self, z_style: np.ndarray, z_skill: np.ndarray,
                                     save_path: str = None) -> Tuple[plt.Figure, Dict[str, float]]:
        """Create orthogonality analysis using OrthogonalityEvaluator logic"""
        # Single plot for correlation heatmap only
        fig, ax = plt.subplots(1, 1, figsize=(3.35, 2.8))

        results = {}

        # Calculate cosine similarity metrics (without plotting)
        similarities = []
        for i in range(len(z_style)):
            # Reshape vectors for cosine similarity calculation
            sim = cosine_similarity(z_style[i].reshape(1, -1), z_skill[i].reshape(1, -1))
            similarities.append(sim[0, 0])

        # Average absolute cosine similarity
        avg_cosine_sim = np.mean(np.abs(similarities))
        results['cosine_similarity'] = avg_cosine_sim

        # Create correlation heatmap
        # Combine style and skill vectors into dataframe
        style_cols = [f'Style_{i}' for i in range(z_style.shape[1])]
        skill_cols = [f'Skill_{i}' for i in range(z_skill.shape[1])]

        df_style = pd.DataFrame(z_style, columns=style_cols)
        df_skill = pd.DataFrame(z_skill, columns=skill_cols)
        combined_df = pd.concat([df_style, df_skill], axis=1)

        # Calculate correlation matrix
        correlation_matrix = combined_df.corr()

        # Extract cross-correlation between style and skill
        cross_correlation = correlation_matrix.loc[style_cols, skill_cols]

        # Create heatmap with compact formatting
        sns.heatmap(cross_correlation, annot=True, cmap='RdBu_r', center=0, fmt='.2f', ax=ax,
                   annot_kws={"size": 6}, cbar_kws={'label': 'Correlation'})

        # Fine-tune tick marks and labels for compact size
        ax.tick_params(axis='both', labelsize=6, width=0.5, length=2)
        ax.tick_params(axis='y', rotation=45)
        # ax.set_title(f'Style-Skill Cross-Correlation\n(Dimensions: {z_style.shape[1]}×{z_skill.shape[1]})', fontsize=16)
        ax.set_xlabel('Skill Dimensions')
        ax.set_ylabel('Style Dimensions')

        # Calculate additional metrics
        max_cross_corr = np.max(np.abs(cross_correlation.values))
        mean_cross_corr = np.mean(np.abs(cross_correlation.values))

        results['max_cross_correlation'] = max_cross_corr
        results['mean_cross_correlation'] = mean_cross_corr

        # Add text summary
        # summary_text = f'Cosine Similarity: {avg_cosine_sim:.4f}\nMax Cross-Corr: {max_cross_corr:.4f}\nMean Cross-Corr: {mean_cross_corr:.4f}'
        # ax.text(1.02, 0.5, summary_text, transform=ax.transAxes, verticalalignment='center',
        #         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=12)

        plt.tight_layout()
        self.save_figure(fig, save_path)
        return fig, results

    def evaluate_model(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
                      device: torch.device, experiment_name: str = "experiment") -> Dict[str, Any]:
        """Complete model evaluation with academic paper quality outputs"""
        print(f"Starting comprehensive academic evaluation for {experiment_name}")

        # Extract data
        print("Extracting latent variables...")
        data = self.extract_latent_variables(model, dataloader, device)

        results = {}

        # 1. Style space analysis
        print("Creating style space visualization...")
        style_fig, style_results = self.create_style_space_visualization(
            data['z_style'], data['subject_ids'],
            save_path=str(self.output_dir / f"{experiment_name}_style_space.pdf")
        )

        # 2. Skill space analysis
        print("Creating skill space visualization...")
        skill_fig = self.create_skill_space_visualization(
            data['z_skill'], data['skill_scores'], data['subject_ids'],
            save_path=str(self.output_dir / f"{experiment_name}_skill_space.pdf")
        )

        # 3. Skill regression analysis
        print("Creating skill regression analysis...")
        regression_fig, regression_results = self.create_skill_regression_analysis(
            data['z_skill'], data['skill_scores'],
            save_path=str(self.output_dir / f"{experiment_name}_skill_regression.pdf")
        )

        # 4. Trajectory analysis
        print("Creating trajectory analysis...")
        trajectory_fig = self.create_trajectory_overlay_analysis(
            data['trajectories'], data['reconstructed'], data['subject_ids'], n_samples=6,
            save_path=str(self.output_dir / f"{experiment_name}_trajectory_overlay.pdf")
        )

        # 5. Orthogonality analysis
        print("Creating orthogonality analysis...")
        orthogonality_fig, orthogonality_results = self.create_orthogonality_analysis(
            data['z_style'], data['z_skill'],
            save_path=str(self.output_dir / f"{experiment_name}_orthogonality.pdf")
        )

        # Calculate comprehensive summary metrics
        reconstruction_mse = np.mean((data['trajectories'] - data['reconstructed'])**2)

        results['summary_metrics'] = {
            'reconstruction_mse': reconstruction_mse,
            'n_subjects': len(set(data['subject_ids'])),
            'n_samples': len(data['trajectories']),
            'style_dimensions': data['z_style'].shape[1],
            'skill_dimensions': data['z_skill'].shape[1],
            'trajectory_length': data['trajectories'].shape[1],
            'feature_dimensions': data['trajectories'].shape[2]
        }

        # Add regression results
        if 'error' not in regression_results:
            results['skill_regression_results'] = regression_results
            results['summary_metrics']['skill_regression_r2'] = regression_results['best_r2']
            results['summary_metrics']['best_regression_method'] = regression_results['best_method']
        else:
            results['skill_regression_results'] = regression_results
            results['summary_metrics']['skill_regression_r2'] = 0.0
            results['summary_metrics']['best_regression_method'] = 'failed'

        # Add orthogonality results
        results['orthogonality_results'] = orthogonality_results
        results['summary_metrics']['style_skill_cosine_similarity'] = orthogonality_results['cosine_similarity']
        results['summary_metrics']['style_skill_max_correlation'] = orthogonality_results['max_cross_correlation']

        print(f"Comprehensive evaluation complete. Results saved to {self.output_dir}")
        print(f"Generated files:")
        print(f"  - {experiment_name}_style_space.pdf/.png")
        print(f"  - {experiment_name}_skill_space.pdf/.png")
        print(f"  - {experiment_name}_skill_regression.pdf/.png")
        print(f"  - {experiment_name}_trajectory_overlay.pdf/.png")
        print(f"  - {experiment_name}_orthogonality.pdf/.png")

        return results