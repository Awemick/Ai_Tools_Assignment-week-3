# ethics_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_model_bias():
    """Analyze potential biases in our models"""
    
    print("ETHICAL ETHICAL AI ANALYSIS")
    print("="*50)
    
    # MNIST Model Bias Analysis
    print("\nFAIRNESS MNIST MODEL BIAS ANALYSIS:")
    mnist_biases = [
        "Cultural Bias: Handwriting styles vary across cultures",
        "Age Bias: Younger vs older handwriting differences", 
        "Educational Bias: Literacy levels affect writing quality",
        "Geographic Bias: Training data primarily from certain regions",
        "Technological Bias: Digital vs traditional writing instruments"
    ]
    
    for i, bias in enumerate(mnist_biases, 1):
        print(f"  {i}. {bias}")
    
    # Amazon Reviews Bias Analysis
    print("\nAMAZON AMAZON REVIEWS MODEL BIAS ANALYSIS:")
    nlp_biases = [
        "Language Bias: Primarily English language focused",
        "Cultural Bias: Western-centric sentiment expressions",
        "Product Bias: Favors popular brands over lesser-known ones",
        "Demographic Bias: Reviews from specific age/income groups",
        "Temporal Bias: Changing language and sentiment patterns over time"
    ]
    
    for i, bias in enumerate(nlp_biases, 1):
        print(f"  {i}. {bias}")
    
    return mnist_biases, nlp_biases

def bias_mitigation_strategies():
    """Propose strategies to mitigate identified biases"""
    
    print("\nMITIGATION BIAS MITIGATION STRATEGIES")
    print("="*50)
    
    strategies = {
        "Data Diversification": [
            "Collect training data from diverse geographic regions",
            "Include various age groups and educational backgrounds",
            "Incorporate multiple languages and cultural contexts"
        ],
        "Algorithmic Fairness": [
            "Implement fairness constraints during training",
            "Use adversarial debiasing techniques",
            "Regular bias auditing with tools like Fairness Indicators"
        ],
        "Evaluation Metrics": [
            "Measure performance across different demographic groups",
            "Track disparate impact and equal opportunity metrics",
            "Regular ethical reviews of model outputs"
        ],
        "Transparency": [
            "Document data sources and limitations",
            "Provide model cards with bias disclosures",
            "Enable user feedback for continuous improvement"
        ]
    }
    
    for strategy, methods in strategies.items():
        print(f"\n{strategy}:")
        for method in methods:
            print(f"  • {method}")

def implement_fairness_metrics():
    """Example implementation of fairness metrics"""
    
    # Simulated performance across different groups
    groups = ['Group A', 'Group B', 'Group C', 'Group D']
    accuracy_scores = [0.92, 0.88, 0.85, 0.91]
    precision_scores = [0.89, 0.86, 0.82, 0.88]
    
    fairness_df = pd.DataFrame({
        'Group': groups,
        'Accuracy': accuracy_scores,
        'Precision': precision_scores
    })
    
    print("\nCHART FAIRNESS METRICS ACROSS GROUPS")
    print("="*40)
    print(fairness_df)
    
    # Calculate fairness disparities
    accuracy_disparity = max(accuracy_scores) - min(accuracy_scores)
    precision_disparity = max(precision_scores) - min(precision_scores)
    
    print(f"\nFairness Disparities:")
    print(f"Accuracy Range: {accuracy_disparity:.3f}")
    print(f"Precision Range: {precision_disparity:.3f}")
    
    # Visualization
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(groups))
    width = 0.35
    
    plt.bar(x - width/2, accuracy_scores, width, label='Accuracy', alpha=0.7)
    plt.bar(x + width/2, precision_scores, width, label='Precision', alpha=0.7)
    
    plt.xlabel('Demographic Groups')
    plt.ylabel('Scores')
    plt.title('Model Performance Across Different Groups')
    plt.xticks(x, groups)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0.8, 1.0)
    
    plt.tight_layout()
    plt.show()
    
    return fairness_df

# Run ethical analysis
if __name__ == "__main__":
    mnist_biases, nlp_biases = analyze_model_bias()
    bias_mitigation_strategies()
    fairness_metrics = implement_fairness_metrics()
    
    print("\nCOMPLETE Ethical analysis complete. Always consider:")
    print("   - Regular bias auditing")
    print("   - Diverse training data collection") 
    print("   - Transparent documentation")
    print("   - Continuous monitoring and improvement")