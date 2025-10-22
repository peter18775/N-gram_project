import itertools
import numpy as np
from src.evaluate import evaluate_model
from src.smoothing import LinearInterpolation

def tune_lambdas(models, dev_data, num_points=5):
    """
    Fine-tune lambda values for linear interpolation using grid search on validation data.
    
    Args:
        models: List of n-gram models ordered from unigram to n-gram
        dev_data: Development/validation dataset
        num_points: Number of points to try for each lambda (grid granularity)
    
    Returns:
        best_lambdas: List of optimal lambda values
        best_perplexity: Perplexity achieved with the optimal lambdas
    """
    # Create grid points between 0 and 1
    points = np.linspace(0, 1, num_points)
    
    # Generate all possible combinations of lambda values
    lambda_combinations = list(itertools.product(points, repeat=len(models)))
    
    # Filter combinations where sum of lambdas ≈ 1
    valid_combinations = [comb for comb in lambda_combinations 
                        if abs(sum(comb) - 1.0) == 0]
    
    if not valid_combinations:
        raise ValueError("No valid lambda combinations found. Try increasing num_points.")
    
    best_perplexity = float('inf')
    best_lambdas = None
    
    # Try each valid combination
    for lambdas in valid_combinations:
        interp_model = LinearInterpolation(models, list(lambdas))
        try:
            perplexity = evaluate_model(interp_model, dev_data)
            
            if perplexity < best_perplexity:
                best_perplexity = perplexity
                best_lambdas = lambdas
                
            print(f"Lambdas: {lambdas}, Perplexity: {perplexity:.2f}")
                
        except Exception as e:
            print(f"Error with lambdas {lambdas}: {str(e)}")
            continue
    
    return list(best_lambdas), best_perplexity