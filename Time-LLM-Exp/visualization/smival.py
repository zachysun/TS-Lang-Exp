import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

sys.path.append(os.path.abspath('../'))
from utils.tsfel import *

class SMIValidator:
    def __init__(self, n_groups=5, n_samples=20, n_points=100):
        self.n_groups = n_groups
        self.n_samples = n_samples
        self.n_points = n_points
        self.feature_extraction_funcs = feature_extraction_functions
    
    def generate_synthetic_data(self):
        """
        Generate synthetic data
        
        Parameters:
            n_groups: Number of groups to generate
            n_samples: Number of samples per group
            n_points: Number of points per sample
            
        Returns:
            Dictionary containing different test scenarios
        """
        n_groups = self.n_groups
        n_samples = self.n_samples
        n_points = self.n_points
        
        synthetic_data = {
            'sintra_sinter': {},
            'sintra_minter': {},
            'sintra_linter': {},
            
            'mintra_sinter': {},
            'mintra_minter': {},
            'mintra_linter': {},
            
            'lintra_sinter': {},
            'lintra_minter': {},
            'lintra_linter': {},
            
            'zero_intra': {},
            'zero_inter': {}
        }
        
        # Define noise levels and phase shifts
        small_noise = 0.03  # Small intra-group variation
        medium_noise = 0.1  # Medium intra-group variation
        large_noise = 0.5   # Large intra-group variation
        
        small_phase = np.pi/10    # Small inter-group variation
        medium_phase = np.pi/5  # Medium inter-group variation
        large_phase = np.pi/2  # Large inter-group variation

        # Small intra-group variation, small inter-group variation
        for i in range(n_groups):
            base = np.sin(np.linspace(0, 4*np.pi, n_points) + i*small_phase) + i * np.random.normal(-small_noise, small_noise, n_points)
            samples = np.array([base + np.random.normal(-small_noise, small_noise, n_points) for _ in range(n_samples)])
            synthetic_data['sintra_sinter'][f'group_{i}'] = samples
            
        # Small intra-group variation, medium inter-group variation
        for i in range(n_groups):
            base = np.sin(np.linspace(0, 4*np.pi, n_points) + i*medium_phase) + i * np.random.normal(-medium_noise, medium_noise, n_points)
            samples = np.array([base + np.random.normal(-small_noise, small_noise, n_points) for _ in range(n_samples)])
            synthetic_data['sintra_minter'][f'group_{i}'] = samples
        
        # Small intra-group variation, large inter-group variation
        for i in range(n_groups):
            base = np.sin(np.linspace(0, 4*np.pi, n_points) + i*large_phase) + i * np.random.normal(-large_noise, large_noise, n_points)
            samples = np.array([base + np.random.normal(-small_noise, small_noise, n_points) for _ in range(n_samples)])
            synthetic_data['sintra_linter'][f'group_{i}'] = samples
            
        # Medium intra-group variation, small inter-group variation
        for i in range(n_groups):
            base = np.sin(np.linspace(0, 4*np.pi, n_points) + i*small_phase) + i * np.random.normal(-small_noise, small_noise, n_points)
            samples = np.array([base + np.random.normal(-medium_noise, medium_noise, n_points) for _ in range(n_samples)])
            synthetic_data['mintra_sinter'][f'group_{i}'] = samples    
        
        # Medium intra-group variation, medium inter-group variation
        for i in range(n_groups):
            base = np.sin(np.linspace(0, 4*np.pi, n_points) + i*medium_phase) + i * np.random.normal(-medium_noise, medium_noise, n_points)
            samples = np.array([base + np.random.normal(-medium_noise, medium_noise, n_points) for _ in range(n_samples)])
            synthetic_data['mintra_minter'][f'group_{i}'] = samples
            
        # Medium intra-group variation, large inter-group variation
        for i in range(n_groups):
            base = np.sin(np.linspace(0, 4*np.pi, n_points) + i*large_phase) + i * np.random.normal(-large_noise, large_noise, n_points)
            samples = np.array([base + np.random.normal(-medium_noise, medium_noise, n_points) for _ in range(n_samples)])
            synthetic_data['mintra_linter'][f'group_{i}'] = samples
            
        # Large intra-group variation, small inter-group variation
        for i in range(n_groups):
            base = np.sin(np.linspace(0, 4*np.pi, n_points) + i*small_phase) + i * np.random.normal(-small_noise, small_noise, n_points)
            samples = np.array([base + np.random.normal(-large_noise, large_noise, n_points) for _ in range(n_samples)])
            synthetic_data['lintra_sinter'][f'group_{i}'] = samples
            
        # Large intra-group variation, medium inter-group variation
        for i in range(n_groups):
            base = np.sin(np.linspace(0, 4*np.pi, n_points) + i*medium_phase) + i * np.random.normal(-medium_noise, medium_noise, n_points)
            samples = np.array([base + np.random.normal(-large_noise, large_noise, n_points) for _ in range(n_samples)])
            synthetic_data['lintra_minter'][f'group_{i}'] = samples
        
        # Large intra-group variation, large inter-group variation
        for i in range(n_groups):
            base = np.sin(np.linspace(0, 4*np.pi, n_points) + i*large_phase) + i * np.random.normal(-large_noise, large_noise, n_points)
            samples = np.array([base + np.random.normal(-large_noise, large_noise, n_points) for _ in range(n_samples)])
            synthetic_data['lintra_linter'][f'group_{i}'] = samples
            
        # Zero intra-group variation
        for i in range(n_groups):
            base = np.sin(np.linspace(0, 4*np.pi, n_points))
            samples = np.array([base * (i + 1) for _ in range(n_samples)])
            synthetic_data['zero_intra'][f'group_{i}'] = samples
            
        # Zero inter-group variation
        base = np.sin(np.linspace(0, 4*np.pi, n_points))
        samples = np.array([base + np.random.normal(-1e-2, 1e-2, n_points) for _ in range(n_samples)])
        for i in range(n_groups):
            synthetic_data['zero_inter'][f'group_{i}'] = samples
            

        return synthetic_data
    
    def cal_semantic_matching_index(self, token_patch_dict):
        """
        Calculate Semantic Matching Index (SMI)
        """
        a = 0.5
        b = 0.1
        
        feature_diff_metrics = {}
        feature_extraction_funcs = self.feature_extraction_funcs

        for token_set, patches in token_patch_dict.items():
            feature_values = {func.__name__: [func(patch) for patch in patches] for func in feature_extraction_funcs}
            token_set_features = {}
            
            for func in feature_extraction_funcs:
                feature_name = func.__name__
                token_set_features[f'mean_{feature_name}'] = np.mean(feature_values[feature_name])
                token_set_features[f'std_{feature_name}'] = np.std(feature_values[feature_name])

            feature_diff_metrics[token_set] = token_set_features

        token_sets = list(token_patch_dict.keys())
        num_token_sets = len(token_sets)

        diff_within_token_set = sum(
            sum(feature_diff_metrics[token_set][f'std_{func.__name__}'] for func in feature_extraction_funcs)
            for token_set in token_sets
        )
        avg_diff_within = diff_within_token_set / num_token_sets

        diff_between_token_set = 0
        comp_count = 0
        for i in range(num_token_sets - 1):
            for j in range(i + 1, num_token_sets):
                diff_sum = sum(
                    abs(feature_diff_metrics[token_sets[i]][f'mean_{func.__name__}'] -
                        feature_diff_metrics[token_sets[j]][f'mean_{func.__name__}'])
                    for func in feature_extraction_funcs
                )
                diff_between_token_set += diff_sum
                comp_count += 1

        # Calculate average difference between token sets
        avg_diff_between = diff_between_token_set / comp_count if comp_count > 0 else 0

        if avg_diff_within < 1e-8:
            semantic_matching_ratio = float('inf')
        else:
            semantic_matching_ratio = a * avg_diff_between / avg_diff_within
        
        semantic_matching_index = 1 - np.exp(-b * semantic_matching_ratio)

        return semantic_matching_index, feature_diff_metrics
    
    def cal_alternative_metrics(self, token_patch_dict):
        """
        Calculate alternative comparison metrics
        """
        # Prepare data
        all_patches = []
        labels = []
        for label, patches in token_patch_dict.items():
            all_patches.extend(patches)
            labels.extend([label] * len(patches))
        all_patches = np.array(all_patches)
        
        # Calculate silhouette score
        silhouette = silhouette_score(all_patches, labels)
        
        return {
            'silhouette_score': silhouette,
        }
    
    def validate(self):
        """Execute complete validation process"""
        # Generate synthetic data
        synthetic_data = self.generate_synthetic_data()
        
        results = {}
        for data_type, token_dict in synthetic_data.items():
            # Calculate SMI
            smi, feature_metrics = self.cal_semantic_matching_index(token_dict)
            # Calculate comparison metrics
            alt_metrics = self.cal_alternative_metrics(token_dict)
            
            results[data_type] = {
                'semantic_matching_index': smi,
                'feature_metrics': feature_metrics,
                **alt_metrics
            }
            
        self.visualize_results(results)
        
        return results
    
    def visualize_results(self, results):
        metrics = [
                    'semantic_matching_index', 
                    'silhouette_score',
                ]
        data_types = list(results.keys())
        
        colors = {
            'sintra': '#CC6666',  
            'mintra': '#66CC66',  
            'lintra': '#6666CC',  
            'zero': '#CCCC66'     
        }
        
        fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 13))
        plt.subplots_adjust(hspace=0.4)
        
        for i, metric in enumerate(metrics):
            values = [results[dt][metric] for dt in data_types]
            
            metric_title = metric.replace('_', ' ').title()
            
            bar_colors = [colors[dt.split('_')[0]] for dt in data_types]
            x_positions = np.arange(len(data_types)) * 0.6
            bars = axes[i].bar(x_positions, values, color=bar_colors, width=0.5)
            
            axes[i].set_title(metric_title, fontsize=20, pad=15)
            axes[i].grid(True, axis='y', linestyle='--', alpha=0.7)
            axes[i].set_xticks(x_positions)
            axes[i].set_xticklabels(data_types, rotation=45, ha='right', fontsize=15)
            
            axes[i].set_ylim(-0.1, max(max(values) * 1.1, 1.0))
            for bar in bars:
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}',
                            ha='center', va='bottom', rotation=0, fontsize=16)
        
        plt.savefig('./results/smi_val_results.png', 
                    dpi=300, 
                    bbox_inches='tight',
                    pad_inches=0.5)
        plt.close()
        
        print("\nValidation Results:")
        for data_type, metrics in results.items():
            print(f"\n{data_type}:")
            print(f"Semantic Matching Index: {metrics['semantic_matching_index']:.4f}")
            print(f"Silhouette Score: {metrics['silhouette_score']:.4f}")

if __name__ == "__main__":
    validator = SMIValidator(n_groups=5, n_samples=20, n_points=100)
    validation_results = validator.validate()
