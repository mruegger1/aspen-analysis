def analyze_feature_importance(best_model, feature_meta, X):
    """Analyze feature importance for tree-based models"""
    logger.info("Analyzing feature importance")
    
    if hasattr(best_model, 'feature_importances_'):
        # Get feature names after preprocessing
        preprocessor = best_model['preprocessor'] if isinstance(best_model, Pipeline) else None
        
        if preprocessor:
            # Training the preprocessor separately to get feature names
            preprocessor.fit(X)
            
            # Get feature names after one-hot encoding
            cat_features = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(
                feature_meta['categorical_features']
            )
            
            # Combine all feature names
            all_features = list(feature_meta['numeric_features']) + \
                            list(feature_meta['geo_features']) + \
                            list(cat_features) + \
                            list(feature_meta['boolean_features'])
            
            # Get feature importances
            importances = best_model.feature_importances_
            
            # Create DataFrame for visualization
            feature_importances = pd.DataFrame({
                'feature': all_features[:len(importances)],
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # Visualize top 20 features
            plt.figure(figsize=(12, 8))
            sns.barplot(x='importance', y='feature', data=feature_importances.head(20))
            plt.title('Top 20 Feature Importance')
            plt.tight_layout()
            plt.savefig('feature_importance.png')
            
            return feature_importances
    else:
        logger.warning("Model doesn't provide feature importances")
        return None