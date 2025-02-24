import pandas as pd
from pathlib import Path
from openai import OpenAI
import time
from typing import Optional, List, Dict

class FeatureAnalyzer:
    def __init__(self):
        self.base_path = Path("/Users/juliaschafer/Downloads/Results/")
        self.client = OpenAI(api_key="add key here"))  
        
        self.dataset_files = {
            'Acevedo': ['Acevedo_0shot_classification_gpt-4o_answers_conf_mat.csv'],
            'Bone_Marrow_Cyto': ['Bone_Marrow_Cyto_0shot_classification_gpt-4o_answers_conf_mat.csv'],
        }

    def get_features(self, true_label: str, predicted_label: str) -> Optional[List[Dict]]:
        prompt = f"""
        For this blood cell classification case where {true_label} was classified as {predicted_label}, analyze the key distinguishing features.

        List exactly 3 specific morphological or structural features that would be most important for an expert to correctly distinguish between these cell types. For each feature, assign a numerical importance score (1-100) reflecting how crucial that feature is for differentiation. The scores must sum to exactly 100.

        Consider these categories of features:
        - Nuclear characteristics (shape, size, chromatin pattern, lobulation)
        - Cytoplasmic features (granularity, color, volume)
        - Overall cell size and shape
        - Membrane characteristics
        - Specific cellular structures or inclusions

        Required format for each line:
        feature_name,importance_score,detailed_explanation

        Key requirements:
        - Feature names must be specific and clear (e.g., 'nuclear_shape' not just 'shape')
        - Importance scores must be decimal numbers that sum to exactly 100
        - Features must be ordered by descending importance
        - Explanations must detail why this feature is specifically relevant for distinguishing {true_label} from {predicted_label}
        - Each feature should be unique and focus on a different aspect of the cell

        Example format:
        nuclear_shape,45.5,The distinctive band-shaped nucleus is the key feature that separates this cell type
        cytoplasmic_granularity,35.0,The specific pattern and density of granules provides strong differentiation
        chromatin_pattern,19.5,The characteristic chromatin distribution helps confirm the cell identity
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000
            )
            
            features = []
            for line in response.choices[0].message.content.strip().split('\n'):
                if line:
                    name, score, explanation = line.split(',', 2)
                    features.append({
                        'name': name.strip(),
                        'importance': float(score),
                        'explanation': explanation.strip()
                    })
            
            total_score = sum(feature['importance'] for feature in features)
            if abs(total_score - 100) > 0.1:  
                print(f"Warning: Feature scores sum to {total_score}, not 100")
            
            return features
            
        except Exception as e:
            print(f"Error getting features: {e}")
            return None

    def analyze_confusion_matrix(self, dataset_name: str, filename: str) -> Optional[pd.DataFrame]:
        conf_mat_file = self.base_path / dataset_name / "confusion_matrices" / filename
        
        print(f"\nAnalyzing {dataset_name} - {filename}")
        
        if not conf_mat_file.exists():
            print(f"File not found: {conf_mat_file}")
            return None

        df = pd.read_csv(conf_mat_file, header=1)
        csv_data = []
        cell_types = [col for col in df.columns[2:-1]]
        processed_pairs = set()
        
        for idx, row in df.iterrows():
            if isinstance(row[0], str) and row[0].strip() == 'cell_type':
                true_label = row[1]
                
                for pred_label in cell_types:
                    count = row[pred_label]
                    if pd.notna(count) and float(count) > 0:
                        pair_key = f"{true_label}-{pred_label}"
                        
                        if pair_key not in processed_pairs:
                            processed_pairs.add(pair_key)
                            print(f"Analyzing: {true_label} -> {pred_label}")
                            
                            features = self.get_features(true_label, pred_label)
                            if features:
                                for i, feature in enumerate(features, 1):
                                    csv_data.append({
                                        'dataset': dataset_name,
                                        'true_label': true_label,
                                        'predicted_label': pred_label,
                                        'feature_rank': i,
                                        'feature_name': feature['name'],
                                        'importance_score': feature['importance'],
                                        'explanation': feature['explanation']
                                    })
                            time.sleep(1)  
        
        if csv_data:
            return pd.DataFrame(csv_data)
        return None

    def run_analysis(self):
        all_results = []
        
        for dataset, files in self.dataset_files.items():
            for filename in files:
                results = self.analyze_confusion_matrix(dataset, filename)
                if results is not None:
                    all_results.append(results)
        
        if all_results:
            final_df = pd.concat(all_results, ignore_index=True)
            
            output_file = self.base_path / 'feature_analysis_results4.csv'
            final_df.to_csv(output_file, index=False)
            print(f"\nDetailed results saved to: {output_file}")
            
            summary = final_df.groupby(['dataset', 'feature_name']).agg({
                'importance_score': ['mean', 'std', 'count']
            }).reset_index()
            
            summary_file = self.base_path / 'feature_analysis_summary4.csv'
            summary.to_csv(summary_file, index=False)
            print(f"Summary statistics saved to: {summary_file}")
            
            return final_df
        
        return None

def main():
    analyzer = FeatureAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main()


