"""
Random Forest Classifier for Tennis Match Prediction
Supports both numpy arrays and pandas DataFrames
"""

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Union
import joblib


class RandomForest:
    """
    Random Forest classifier with bagging and subspace sampling for tennis match prediction.
    Accepts pre-split train/test data (both numpy arrays and pandas DataFrames).
    """
    
    def __init__(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
        n_estimators: int = 2000,
        max_depth: int = 15,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        max_samples: float = 0.8,
        max_features: float = 0.8,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        """
        Initialize RandomForest classifier with pre-split data and hyperparameters.
        
        Parameters:
        -----------
        X_train : pd.DataFrame or np.ndarray
            Training features
        y_train : pd.Series or np.ndarray
            Training target
        X_test : pd.DataFrame or np.ndarray
            Test features
        y_test : pd.Series or np.ndarray
            Test target
        n_estimators : int, default=200
            Number of trees in the forest
        max_depth : int, default=15
            Maximum depth of each tree
        min_samples_split : int, default=5
            Minimum samples required to split a node
        min_samples_leaf : int, default=2
            Minimum samples required in a leaf node
        max_samples : float, default=0.8
            Fraction of samples to use for each tree (bagging)
        max_features : float, default=0.8
            Fraction of features for subspace sampling
        random_state : int, default=42
            Random seed for reproducibility
        n_jobs : int, default=-1
            Number of parallel jobs (-1 uses all processors)
        """
        # Hyperparameters
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_samples = max_samples
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        # Initialize model
        self.model = None
        
        # Store data - handle both DataFrame and numpy array
        if isinstance(X_train, pd.DataFrame):
            self.X_train = X_train.copy()
            self.X_test = X_test.copy()
        else:
            self.X_train = X_train
            self.X_test = X_test
        
        if isinstance(y_train, pd.Series):
            self.y_train = y_train.copy()
            self.y_test = y_test.copy()
        else:
            self.y_train = y_train
            self.y_test = y_test
        
        # Validate and convert data if needed
        self._validate_data()
        
        # Predictions storage
        self.y_train_pred = None
        self.y_test_pred = None
        self.y_train_proba = None
        self.y_test_proba = None
        
        # Metrics storage
        self.metrics = {}
        self.cv_results = {}
        
        # Print data info
        self._print_data_info()
        
    def _validate_data(self) -> None:
        """
        Validate input data and check for common issues.
        Converts numpy arrays to DataFrames for easier handling.
        """
        # Convert numpy arrays to DataFrame if needed
        if isinstance(self.X_train, np.ndarray):
            n_features = self.X_train.shape[1]
            feature_names = [f"feature_{i}" for i in range(n_features)]
            self.X_train = pd.DataFrame(self.X_train, columns=feature_names)
            self.X_test = pd.DataFrame(self.X_test, columns=feature_names)
            print(f"✓ Converted numpy arrays to DataFrames with {n_features} features")
        
        # Convert numpy arrays to Series if needed (for y)
        if isinstance(self.y_train, np.ndarray):
            self.y_train = pd.Series(self.y_train)
            self.y_test = pd.Series(self.y_test)
        
        # Check if data is not empty
        if len(self.X_train) == 0 or len(self.X_test) == 0:
            raise ValueError("Training or test set is empty!")
        
        # Check if features match
        if list(self.X_train.columns) != list(self.X_test.columns):
            raise ValueError("Training and test sets have different features!")
        
        # Check for NaN values
        if self.X_train.isnull().any().any():
            n_nan = self.X_train.isnull().sum().sum()
            print(f"⚠️  Warning: {n_nan} NaN values found in training features. Filling with 0...")
            self.X_train = self.X_train.fillna(0)
            
        if self.X_test.isnull().any().any():
            n_nan = self.X_test.isnull().sum().sum()
            print(f"⚠️  Warning: {n_nan} NaN values found in test features. Filling with 0...")
            self.X_test = self.X_test.fillna(0)
        
        # Check for infinite values
        if np.isinf(self.X_train.values).any():
            print("⚠️  Warning: Infinite values found in training set. Replacing with 0...")
            self.X_train = self.X_train.replace([np.inf, -np.inf], 0)
            
        if np.isinf(self.X_test.values).any():
            print("⚠️  Warning: Infinite values found in test set. Replacing with 0...")
            self.X_test = self.X_test.replace([np.inf, -np.inf], 0)
        
        # Check if all features are numeric
        non_numeric_train = self.X_train.select_dtypes(exclude=[np.number]).columns
        non_numeric_test = self.X_test.select_dtypes(exclude=[np.number]).columns
        
        if len(non_numeric_train) > 0 or len(non_numeric_test) > 0:
            raise ValueError(f"Non-numeric features found! Train: {list(non_numeric_train)}, Test: {list(non_numeric_test)}")
        
        print("✓ Data validation completed successfully")
        
    def _print_data_info(self) -> None:
        """
        Print information about the loaded data.
        """
        print("\n" + "=" * 70)
        print("DATA INFORMATION")
        print("=" * 70)
        
        # Get number of features
        if isinstance(self.X_train, pd.DataFrame):
            n_features = len(self.X_train.columns)
        else:
            n_features = self.X_train.shape[1]
        
        print(f"Training Set:")
        print(f"  Samples:                {len(self.X_train)}")
        print(f"  Features:               {n_features}")
        print(f"  Class Distribution:")
        print(f"    Class 0 (Player 2):   {(self.y_train == 0).sum()} ({(self.y_train == 0).mean()*100:.2f}%)")
        print(f"    Class 1 (Player 1):   {(self.y_train == 1).sum()} ({(self.y_train == 1).mean()*100:.2f}%)")
        
        print(f"\nTest Set:")
        print(f"  Samples:                {len(self.X_test)}")
        print(f"  Features:               {n_features}")
        print(f"  Class Distribution:")
        print(f"    Class 0 (Player 2):   {(self.y_test == 0).sum()} ({(self.y_test == 0).mean()*100:.2f}%)")
        print(f"    Class 1 (Player 1):   {(self.y_test == 1).sum()} ({(self.y_test == 1).mean()*100:.2f}%)")
        
        print(f"\nTotal Samples:            {len(self.X_train) + len(self.X_test)}")
        print(f"Train/Test Split:         {len(self.X_train)/(len(self.X_train)+len(self.X_test))*100:.2f}% / {len(self.X_test)/(len(self.X_train)+len(self.X_test))*100:.2f}%")
        print("=" * 70 + "\n")
        
    def build_model(self) -> None:
        """
        Build the Bagging Random Forest model with specified hyperparameters.
        """
        self.model = BaggingClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features='sqrt'
            ),
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            max_features=self.max_features,
            bootstrap=True,
            bootstrap_features=False,
            oob_score=True,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=1
        )
        print("✓ Model built successfully")
        
    def train(self) -> None:
        """
        Train the Random Forest model on training data.
        """
        if self.model is None:
            self.build_model()
        
        print("\n" + "=" * 70)
        print("TRAINING MODEL")
        print("=" * 70)
        print("Training Bagging Random Forest model...")
        
        self.model.fit(self.X_train, self.y_train)
        
        print("✓ Training completed!\n")
        
    def predict(self) -> None:
        """
        Generate predictions on training and test sets.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Predictions on training and test sets
        self.y_train_pred = self.model.predict(self.X_train)
        self.y_test_pred = self.model.predict(self.X_test)
        
        # Probability predictions for ROC-AUC
        self.y_train_proba = self.model.predict_proba(self.X_train)[:, 1]
        self.y_test_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        print("✓ Predictions generated")
        
    def calculate_metrics(self) -> Dict:
        """
        Calculate all performance metrics.
        
        Returns:
        --------
        Dict containing all computed metrics
        """
        if self.y_test_pred is None:
            raise ValueError("Predictions not generated. Call predict() first.")
        
        print("\n" + "=" * 70)
        print("PERFORMANCE METRICS")
        print("=" * 70)
        
        # 1. Accuracy Metrics
        print("\n1. ACCURACY METRICS")
        print("-" * 70)
        train_accuracy = accuracy_score(self.y_train, self.y_train_pred)
        test_accuracy = accuracy_score(self.y_test, self.y_test_pred)
        oob_score = self.model.oob_score_
        
        print(f"Training Accuracy:        {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
        print(f"Test Accuracy:            {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"Out-of-Bag (OOB) Score:   {oob_score:.4f} ({oob_score*100:.2f}%)")
        print(f"Overfitting Gap:          {(train_accuracy - test_accuracy):.4f}")
        
        # 2. Precision, Recall, F1-Score
        print("\n2. PRECISION, RECALL, F1-SCORE")
        print("-" * 70)
        
        train_precision = precision_score(self.y_train, self.y_train_pred)
        train_recall = recall_score(self.y_train, self.y_train_pred)
        train_f1 = f1_score(self.y_train, self.y_train_pred)
        
        test_precision = precision_score(self.y_test, self.y_test_pred)
        test_recall = recall_score(self.y_test, self.y_test_pred)
        test_f1 = f1_score(self.y_test, self.y_test_pred)
        
        print(f"                    Training Set    Test Set")
        print(f"Precision:          {train_precision:.4f}          {test_precision:.4f}")
        print(f"Recall:             {train_recall:.4f}          {test_recall:.4f}")
        print(f"F1-Score:           {train_f1:.4f}          {test_f1:.4f}")
        
        # 3. ROC-AUC Score
        print("\n3. ROC-AUC SCORE")
        print("-" * 70)
        train_roc_auc = roc_auc_score(self.y_train, self.y_train_proba)
        test_roc_auc = roc_auc_score(self.y_test, self.y_test_proba)
        
        print(f"Training ROC-AUC:         {train_roc_auc:.4f}")
        print(f"Test ROC-AUC:             {test_roc_auc:.4f}")
        
        # 4. Average Precision (PR-AUC)
        print("\n4. PRECISION-RECALL AUC")
        print("-" * 70)
        train_pr_auc = average_precision_score(self.y_train, self.y_train_proba)
        test_pr_auc = average_precision_score(self.y_test, self.y_test_proba)
        
        print(f"Training PR-AUC:          {train_pr_auc:.4f}")
        print(f"Test PR-AUC:              {test_pr_auc:.4f}")
        
        # 5. Confusion Matrix
        print("\n5. CONFUSION MATRIX (Test Set)")
        print("-" * 70)
        cm = confusion_matrix(self.y_test, self.y_test_pred)
        print(f"\n                Predicted")
        print(f"                0       1")
        print(f"Actual  0     {cm[0,0]:5d}   {cm[0,1]:5d}")
        print(f"        1     {cm[1,0]:5d}   {cm[1,1]:5d}")
        
        tn, fp, fn, tp = cm.ravel()
        print(f"\nTrue Negatives:  {tn}  (Correctly predicted Player_2 wins)")
        print(f"False Positives: {fp}  (Predicted Player_1 wins, but Player_2 won)")
        print(f"False Negatives: {fn}  (Predicted Player_2 wins, but Player_1 won)")
        print(f"True Positives:  {tp}  (Correctly predicted Player_1 wins)")
        
        # 6. Classification Report
        print("\n6. DETAILED CLASSIFICATION REPORT (Test Set)")
        print("-" * 70)
        print(classification_report(self.y_test, self.y_test_pred, 
                                  target_names=['Player_2_wins', 'Player_1_wins'],
                                  digits=4))
        
        # Store metrics
        self.metrics = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'oob_score': oob_score,
            'train_precision': train_precision,
            'test_precision': test_precision,
            'train_recall': train_recall,
            'test_recall': test_recall,
            'train_f1': train_f1,
            'test_f1': test_f1,
            'train_roc_auc': train_roc_auc,
            'test_roc_auc': test_roc_auc,
            'train_pr_auc': train_pr_auc,
            'test_pr_auc': test_pr_auc,
            'confusion_matrix': cm
        }
        
        return self.metrics
        
    def cross_validate(self, n_splits: int = 5) -> Dict:
        """
        Perform time-series cross-validation.
        
        Parameters:
        -----------
        n_splits : int, default=5
            Number of splits for cross-validation
            
        Returns:
        --------
        Dict containing cross-validation results
        """
        print("\n7. TIME-SERIES CROSS-VALIDATION")
        print("-" * 70)
        
        # Combine train and test for CV (maintaining temporal order)
        X_all = pd.concat([self.X_train, self.X_test], ignore_index=True)
        y_all = pd.concat([self.y_train, self.y_test], ignore_index=True)
        
        # Use TimeSeriesSplit for time-aware cross-validation
        test_size = max(len(self.X_test) // n_splits, 1)
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
        
        # Multiple scoring metrics
        scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        print("Performing time-series cross-validation (this may take a while)...")
        for metric in scoring_metrics:
            cv_scores = cross_val_score(self.model, X_all, y_all, cv=tscv, 
                                        scoring=metric, n_jobs=self.n_jobs)
            self.cv_results[metric] = cv_scores
            print(f"{metric.upper():12s}: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return self.cv_results
        
    def plot_metrics(self, save_path: Optional[str] = None) -> None:
        """
        Generate visualization plots for model performance.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the figure. If None, displays the plot.
        """
        if self.y_test_pred is None:
            raise ValueError("Predictions not generated. Call predict() first.")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Confusion Matrix Heatmap
        cm = self.metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Player_2_wins', 'Player_1_wins'],
                    yticklabels=['Player_2_wins', 'Player_1_wins'],
                    ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix (Test Set)', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Actual')
        axes[0, 0].set_xlabel('Predicted')
        
        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(self.y_test, self.y_test_proba)
        test_roc_auc = self.metrics['test_roc_auc']
        axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, 
                        label=f'ROC curve (AUC = {test_roc_auc:.4f})')
        axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                        label='Random Classifier')
        axes[0, 1].set_xlim([0.0, 1.0])
        axes[0, 1].set_ylim([0.0, 1.05])
        axes[0, 1].set_xlabel('False Positive Rate', fontsize=12)
        axes[0, 1].set_ylabel('True Positive Rate', fontsize=12)
        axes[0, 1].set_title('ROC Curve (Test Set)', fontsize=14, fontweight='bold')
        axes[0, 1].legend(loc="lower right")
        axes[0, 1].grid(alpha=0.3)
        
        # 3. Precision-Recall Curve
        precision_curve, recall_curve, _ = precision_recall_curve(self.y_test, self.y_test_proba)
        test_pr_auc = self.metrics['test_pr_auc']
        axes[1, 0].plot(recall_curve, precision_curve, color='green', lw=2,
                        label=f'PR curve (AUC = {test_pr_auc:.4f})')
        axes[1, 0].set_xlim([0.0, 1.0])
        axes[1, 0].set_ylim([0.0, 1.05])
        axes[1, 0].set_xlabel('Recall', fontsize=12)
        axes[1, 0].set_ylabel('Precision', fontsize=12)
        axes[1, 0].set_title('Precision-Recall Curve (Test Set)', fontsize=14, fontweight='bold')
        axes[1, 0].legend(loc="lower left")
        axes[1, 0].grid(alpha=0.3)
        
        # 4. Cross-Validation Scores Comparison
        if self.cv_results:
            scoring_metrics = list(self.cv_results.keys())
            cv_means = [self.cv_results[metric].mean() for metric in scoring_metrics]
            cv_stds = [self.cv_results[metric].std() for metric in scoring_metrics]
            x_pos = np.arange(len(scoring_metrics))
            
            axes[1, 1].bar(x_pos, cv_means, yerr=cv_stds, align='center', 
                           alpha=0.7, capsize=10, color='steelblue')
            axes[1, 1].set_ylabel('Score', fontsize=12)
            axes[1, 1].set_xticks(x_pos)
            axes[1, 1].set_xticklabels([m.upper() for m in scoring_metrics], rotation=45)
            axes[1, 1].set_title('Time-Series Cross-Validation Scores', 
                                 fontsize=14, fontweight='bold')
            axes[1, 1].set_ylim([0, 1.0])
            axes[1, 1].grid(axis='y', alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Run cross_validate() first', 
                           ha='center', va='center', fontsize=14)
            axes[1, 1].set_title('Cross-Validation Scores', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Plot saved to: {save_path}")
        else:
            plt.show()
        
    def print_summary(self) -> None:
        """
        Print a comprehensive summary of the model and its performance.
        """
        print("\n" + "=" * 70)
        print("MODEL SUMMARY")
        print("=" * 70)
        print(f"Model Type:               Bagging Random Forest")
        print(f"Number of Estimators:     {self.n_estimators}")
        print(f"Max Samples per Tree:     {self.max_samples}")
        print(f"Max Features per Tree:    {self.max_features}")
        print(f"Bootstrap Sampling:       True")
        print(f"Base Estimator:           Decision Tree")
        print(f"  - Max Depth:            {self.max_depth}")
        print(f"  - Min Samples Split:    {self.min_samples_split}")
        print(f"  - Feature Sampling:     sqrt")
        print(f"\nData Split:")
        print(f"  Training Samples:       {len(self.X_train)}")
        print(f"  Test Samples:           {len(self.X_test)}")
        print(f"  Number of Features:     {len(self.X_train.columns)}")
        
        if self.metrics:
            test_roc_auc = self.metrics['test_roc_auc']
            test_f1 = self.metrics['test_f1']
            test_accuracy = self.metrics['test_accuracy']
            print(f"\nPerformance:")
            print(f"  Test Accuracy:          {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
            print(f"  Test F1-Score:          {test_f1:.4f}")
            print(f"  Test ROC-AUC:           {test_roc_auc:.4f}")
            print(f"  OOB Score:              {self.metrics['oob_score']:.4f}")
        
        print("=" * 70)
        
    def run_full_pipeline(
        self,
        run_cv: bool = True,
        plot_results: bool = True,
        save_plot_path: Optional[str] = 'bagging_rf_metrics.png'
    ) -> Dict:
        """
        Run the complete pipeline: build, train, predict, evaluate.
        
        Parameters:
        -----------
        run_cv : bool, default=True
            Whether to run cross-validation
        plot_results : bool, default=True
            Whether to generate plots
        save_plot_path : str, optional
            Path to save plots
            
        Returns:
        --------
        Dict containing all metrics
        """
        # Step 1: Build model
        self.build_model()
        
        # Step 2: Train model
        self.train()
        
        # Step 3: Generate predictions
        self.predict()
        
        # Step 4: Calculate metrics
        self.calculate_metrics()
        
        # Step 5: Cross-validation (optional)
        if run_cv:
            self.cross_validate()
        
        # Step 6: Plot results (optional)
        if plot_results:
            self.plot_metrics(save_plot_path)
        
        # Step 7: Print summary
        self.print_summary()
        
        return self.metrics
        
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        joblib.dump(self.model, filepath)
        print(f"✓ Model saved to: {filepath}")
        
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from disk.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
        """
        self.model = joblib.load(filepath)
        print(f"✓ Model loaded from: {filepath}")


# ============================================
# Usage Example
# ============================================

if __name__ == "__main__":
    print("Random Forest Classifier for Tennis Match Prediction")
    print("=" * 70)
    print("\nUsage:")
    print("  from random_forest_train import RandomForest")
    print("  rf = RandomForest(X_train, y_train, X_test, y_test)")
    print("  metrics = rf.run_full_pipeline()")
