import pandas as pd
from pathlib import Path
from openai import OpenAI
import time

class FeatureAnalyzer:
    def __init__(self):
        self.base_path = Path("cluster path here: (/Users/juliaschafer/Downloads/Results/Acevedo/confusion_matrices)")
        self.client = OpenAI(api_key="add key here")
        
        self.categories = {
            'morphological': ['shape', 'size', 'nucleus', 'cytoplasm', 'granules', 'texture'],
            'context_dependent': ['location', 'arrangement', 'pattern', 'surrounding'],
            'quantitative': ['count', 'ratio', 'number', 'measurement']
        }

    def get_features(self, true_label, predicted_label):
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

        Example format:
        nuclear_chromatin_pattern,95,Dense chromatin pattern unique to this cell type vs the fine chromatin in the other
        cytoplasmic_granularity,85,Presence of specific azurophilic granules distinguishes from agranular types
        nuclear_segmentation,80,Multi-lobed nucleus characteristic of mature forms vs round nucleus
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o, or other model",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            features = []
            for line in response.choices[0].message.content.strip().split('\n'):
                if line:
                    name, score, explanation = line.split(',', 2)
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

    def categorize_feature(self, feature_name):
        feature_name = feature_name.lower()
        for category, keywords in self.categories.items():
            if any(keyword in feature_name for keyword in keywords):
                return category
        return 'other'

    def analyze_and_save_results(self, conf_mat_file):
        print(f"Reading confusion matrix from: {conf_mat_file}")
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
        
        output_file = self.base_path.parent / 'feature_analysis_results.csv'
        results_df.to_csv(output_file, index=False)
        print(f"\nDetailed results saved to: {output_file}")
        
        summary_data = []
        for category in self.categories.keys() | {'other'}:
            category_scores = results_df[results_df['feature_category'] == category]['importance_score']
            if len(category_scores) > 0:
                summary_data.append({
                    'category': category,
                    'avg_importance': category_scores.mean(),
                    'count': len(category_scores),
                    'std_dev': category_scores.std()
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = self.base_path.parent / 'feature_category_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        print(f"Category summary saved to: {summary_file}")

def main():
    analyzer = FeatureAnalyzer()
    base_file = analyzer.base_path / "add confusion matrix here: f.e.: Acevedo_0shot_classification_gpt-4o_answers_conf_mat.csv"
    
    if not base_file.exists():
        print(f"File not found: {base_file}")
        return
    
    analyzer.analyze_and_save_results(base_file)

if __name__ == "__main__":
    main()
