import numpy as np

def calculate_finger_similarity(detected_pattern, canonical_patterns):
    """
    Calculate similarity between detected finger pattern and all canonical patterns
    Returns best match with similarity score
    """
    detected = np.array(detected_pattern)
    similarities = {}
    
    for gesture, canonical in canonical_patterns.items():
        canonical_arr = np.array(canonical)
        
        # Hamming distance (number of different fingers)
        diff_count = np.sum(detected != canonical_arr)
        
        # Similarity score: 1.0 = perfect match, decreases with more differences
        similarity = 1.0 - (diff_count / len(detected))
        
        similarities[gesture] = {
            'similarity': similarity,
            'diff_count': diff_count,
            'canonical': canonical
        }
    
    # Sort by similarity
    sorted_similarities = sorted(similarities.items(), 
                               key=lambda x: x[1]['similarity'], 
                               reverse=True)
    
    return sorted_similarities


def enhance_prediction_with_finger_matching(model_prediction, model_confidence, 
                                          detected_fingers, canonical_patterns):
    """
    Enhance ML prediction with finger similarity matching
    """
    finger_similarities = calculate_finger_similarity(detected_fingers, canonical_patterns)
    
    print(f"\n=== FINGER SIMILARITY ANALYSIS ===")
    print(f"Detected fingers: {detected_fingers}")
    print(f"ML Prediction: {model_prediction} ({model_confidence:.3f})")
    
    print(f"\nFinger similarities:")
    for gesture, data in finger_similarities[:3]:  # Top 3
        print(f"  {gesture}: {data['similarity']:.3f} (diff: {data['diff_count']}, canonical: {data['canonical']})")
    
    # Get best finger match
    best_finger_match, best_finger_data = finger_similarities[0]
    
    # Decision logic
    if best_finger_data['diff_count'] <= 1 and best_finger_data['similarity'] >= 0.8:
        if best_finger_match != model_prediction:
            print(f"\n🔄 FINGER OVERRIDE: {model_prediction} → {best_finger_match}")
            print(f"   Reason: Only {best_finger_data['diff_count']} finger different, high similarity ({best_finger_data['similarity']:.3f})")
            return best_finger_match, f"finger_corrected_{best_finger_data['similarity']:.3f}"
        else:
            print(f"✅ FINGER CONFIRMS: ML and finger matching agree")
            return model_prediction, f"confirmed_{model_confidence:.3f}"
    else:
        print(f"⚠️  FINGER UNCERTAIN: Too many differences ({best_finger_data['diff_count']})")
        return model_prediction, f"ml_only_{model_confidence:.3f}"


# Example usage
if __name__ == "__main__":
    # Define canonical patterns (from your training data)
    canonical_patterns = {
        'next_slide': [0,1,0,0,0],      # Right hand only
        'previous_slide': [0,1,0,0,0],  # Right hand only  
        'zoom_in': [1,1,0,0,0],         # Right hand only
        'zoom_out': [1,1,0,0,0],        # Right hand only
    }
    
    # Test case from your example
    detected = [0,1,1,0,0]  # What was detected
    ml_prediction = "previous_slide"
    ml_confidence = 0.973
    
    final_prediction, final_confidence = enhance_prediction_with_finger_matching(
        ml_prediction, ml_confidence, detected, canonical_patterns
    )