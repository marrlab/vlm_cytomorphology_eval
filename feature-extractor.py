import pandas as pd
from pathlib import Path
from openai import OpenAI
import time

class FeatureAnalyzer:
    def __init__(self):
        self.base_path = Path("cluster path here: (/Users/juliaschafer/Downloads/Results/)")
        self.client = OpenAI(api_key="add key here")
        
        self.categories = {
            'morphological': ['shape', 'size', 'nucleus', 'cytoplasm', 'granules', 'texture'],
            'context_dependent': ['location', 'arrangement', 'pattern', 'surrounding'],
            'quantitative': ['count', 'ratio', 'number', 'measurement']
        }
        self.dataset_files {
            'Acevedo': ['Acevedo_0shot_classification_gpt-4o_answers_conf_mat.csv'],
            'Bone_Marrow_Cyto': ['Bone_Marrow_Cyto_0shot_classification_gpt-4o_answers_conf_mat.csv'],
            'AML_Matek': ['AML_Matek_0shot_classification_gpt-4o_answers_conf_mat.csv',
                         'AML_Matek_0shot_classification_gpt-4o_answers_reviewed_gpt-4o_conf_mat.csv']
                         
                        }


    def get_features(self, true_label: str, predicted_label: str) -> Optional[List[Dict]]:
        prompt = f"""
        For this blood cell classification task where {true_label} was classified as {predicted_label}:

        List exactly 3 specific morphological or structural features that are most important for distinguishing between these cell types. Consider features like:
        - Nuclear characteristics (size, shape, chromatin pattern)
        - Cytoplasmic features (color, granularity, amount)
        - Overall cell size and shape
        - Presence or absence of specific structures

        Format your response strictly as 3 lines:
        feature_name,importance_score(0-100),detailed_explanation

        Guidelines:
        - Feature_name should be specific (e.g., 'nuclear_size' not just 'size')
        - Importance_score should reflect how crucial this feature is for distinguishing THESE SPECIFIC cell types
        - Explanation should detail why this feature helps distinguish between these specific cell types
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            features = []
            for line in response.choices[0].message.content.strip().split('\n'):
                if line:
                    name, score, explanation = line.split(',',2)
                    feature_category = self.categorize_feature(name.strip().lower())
                    features.append({
                        'name': name.strip(),
                        'category': feature_category,
                        'importance': float(score),
                        'explanation': explanation.strip()
                    })
            return features
            
        except Exception as e:
            print(f"Error getting features: {e}")
            return None

    def categorize_feature(self, feature_name: str): -> str:
        
        feature_name = feature_name.lower()
        for category, keywords in self.categories.items():
            if any(keyword in feature_name for keyword in keywords):
                return category
        return 'other'

     def get_confusion_matrix_path(self, dataset_name: str, filename: str) -> Path:
        return self.base_path / dataset_name / "confusion_matrices" / filename
         
    def analyze_and_save_results(self, conf_mat_file):
        print(f"Reading confusion matrix from: {conf_mat_file}")
        conf_mat_file = self.get_confusion_matrix_path(dataset_name, filename)
        
        print(f"\nAnalyzing {dataset_name} - {filename}")
        print(f"Reading confusion matrix from: {conf_mat_file}")
        
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
                                        'true_label': true_label,
                                        'predicted_label': pred_label,
                                        'feature_rank': i,
                                        'feature_name': feature['name'],
                                        'feature_category': feature['category'],
                                        'importance_score': feature['importance'],
                                        'explanation': feature['explanation']
                                    })
                            time.sleep(1)
        
        results_df = pd.DataFrame(csv_data)

    def run_analysis(self):
        all_results = []
       
        for dataset, files in self.dataset_files.items():
            for filename in files:
                results = self.analyze_confusion_matrix(dataset, filename)
                if results is not None:
                    all_results.append(results)
        
        if all_results:
            final_df = pd.concat(all_results, ignore_index=True)
            
            output_file = self.base_path / 'feature_analysis_results.csv'
            final_df.to_csv(output_file, index=False)
            print(f"\nDetailed results saved to: {output_file}")
            
            summary = final_df.groupby(['dataset', 'feature_category']).agg({
                'importance_score': ['mean', 'std', 'count']
            }).reset_index()
            
            summary_file = self.base_path / 'feature_category_summary.csv'
            summary.to_csv(summary_file, index=False)
            print(f"Category summary saved to: {summary_file}")
            
            return final_df
        
        return None
        
def main():
    analyzer = FeatureAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
