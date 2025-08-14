#!/usr/bin/env python

import numpy as np
import logging
from typing import Dict, List
from datasets import DatasetDict
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import RDLogger
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


class ECFP4XGBoostBaseline:
    """
    ECFP4 + XGBoost baseline for molecular property prediction.
    
    This class implements a common cheminformatics baseline that uses:
    - ECFP4 (Extended Connectivity Fingerprints with radius 2) for molecular representation
    - XGBoost for regression
    - Grid search for hyperparameter optimization
    """
    
    def __init__(self, radius: int = 2, n_bits: int = 2048, optimize_hyperparams: bool = True):
        """
        Initialize the ECFP4+XGBoost baseline.
        
        Args:
            radius: Radius for ECFP fingerprints (default: 2 for ECFP4)
            n_bits: Number of bits in the fingerprint (default: 2048)
            optimize_hyperparams: Whether to do limited hyperparameter optimization (default: True)
        """
        self.radius = radius
        self.n_bits = n_bits
        self.optimize_hyperparams = optimize_hyperparams
        

        self.mfpgen = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.radius, 
            fpSize=self.n_bits
        )
        
        self.param_grid = [
            {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.1},
            {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1},
            {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.1},
            {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1},
            {'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.05},
            {'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.05},
            {'n_estimators': 50, 'max_depth': 6, 'learning_rate': 0.2},
            {'n_estimators': 50, 'max_depth': 8, 'learning_rate': 0.2},
        ]
        
        self.fixed_params = {
            'subsample': 1.0,
            'colsample_bytree': 1.0,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse'
        }
        
        self.default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            **self.fixed_params
        }
    
    def smiles_to_ecfp4(self, smiles_list: List[str]) -> np.ndarray:
        """
        Convert SMILES strings to ECFP4 fingerprints.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            NumPy array of shape (n_molecules, n_bits) containing fingerprints
            
        Raises:
            ValueError: If any SMILES string is invalid or cannot be processed
        """
        fingerprints = []
        
        for i, smi in enumerate(smiles_list):
            if smi is None or not isinstance(smi, str):
                raise ValueError(f"Invalid SMILES input at index {i}: {smi} (expected string, got {type(smi)})")
            
            if not smi.strip():
                raise ValueError(f"Empty SMILES string at index {i}")
            
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    raise ValueError(f"Invalid SMILES at index {i}: '{smi}'")
                
                fp = self.mfpgen.GetFingerprintAsNumPy(mol)
                fingerprints.append(fp)
                
            except Exception as e:
                if isinstance(e, ValueError):
                    raise
                else:
                    raise ValueError(f"Error processing SMILES '{smi}' at index {i}: {str(e)}")
        
        return np.array(fingerprints)
    
    def train_and_predict(self, dataset_dict: DatasetDict) -> Dict[str, List[float]]:
        """
        Train XGBoost model on ECFP4 features and make predictions.
        
        Args:
            dataset_dict: DatasetDict containing 'train', 'validation', and 'test' splits
            
        Returns:
            Dictionary containing predictions and labels for the test set
        """

        required_splits = ["train", "validation", "test"]
        for split in required_splits:
            if split not in dataset_dict:
                raise ValueError(f"Dataset must contain a '{split}' split")
        

        train_smiles = dataset_dict["train"]["smiles"]
        train_labels = np.array(dataset_dict["train"]["labels"])
        
        val_smiles = dataset_dict["validation"]["smiles"]
        val_labels = np.array(dataset_dict["validation"]["labels"])
        
        test_smiles = dataset_dict["test"]["smiles"]
        test_labels = np.array(dataset_dict["test"]["labels"])
        

        logging.info("Converting SMILES to ECFP4 fingerprints...")
        train_fps = self.smiles_to_ecfp4(train_smiles)
        val_fps = self.smiles_to_ecfp4(val_smiles)
        test_fps = self.smiles_to_ecfp4(test_smiles)
        

        combined_fps = np.vstack([train_fps, val_fps])
        combined_labels = np.concatenate([train_labels, val_labels])
        

        if self.optimize_hyperparams:
            logging.info(f"Optimizing XGBoost hyperparameters with simple grid search ({len(self.param_grid)} combinations)...")
            
            best_score = float('inf')
            best_params = None
            best_model = None
            
            for i, params in enumerate(self.param_grid):
                full_params = {**params, **self.fixed_params}
                
                model = xgb.XGBRegressor(**full_params)
                model.fit(train_fps, train_labels)
                
                val_predictions = model.predict(val_fps)
                val_score = mean_squared_error(val_labels, val_predictions)
                
                logging.info(f"  Try {i+1}/{len(self.param_grid)}: "
                           f"n_est={params['n_estimators']}, "
                           f"depth={params['max_depth']}, "
                           f"lr={params['learning_rate']}, "
                           f"val_mse={val_score:.4f}")
                
                if val_score < best_score:
                    best_score = val_score
                    best_params = full_params
                    best_model = model
            
            logging.info(f"Best XGBoost parameters: n_estimators={best_params['n_estimators']}, "
                        f"max_depth={best_params['max_depth']}, "
                        f"learning_rate={best_params['learning_rate']}, "
                        f"val_mse={best_score:.4f}")
            
            final_model = xgb.XGBRegressor(**best_params)
            final_model.fit(combined_fps, combined_labels)
            best_model = final_model
            
        else:
            logging.info("Training XGBoost with fixed hyperparameters...")
            best_model = xgb.XGBRegressor(**self.default_params)
            best_model.fit(combined_fps, combined_labels)
            
            logging.info(f"XGBoost parameters used: n_estimators={self.default_params['n_estimators']}, "
                        f"max_depth={self.default_params['max_depth']}, "
                        f"learning_rate={self.default_params['learning_rate']}")
        

        test_predictions = best_model.predict(test_fps)
        

        return {
            "predictions": test_predictions.tolist(),
            "labels": test_labels.tolist()
        }
    
    def get_feature_importance(self, model) -> Dict[str, float]:
        """
        Get feature importance from trained XGBoost model.
        
        Args:
            model: Trained XGBoost model
            
        Returns:
            Dictionary with feature importance information
        """
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            return {
                'mean_importance': float(np.mean(importance)),
                'std_importance': float(np.std(importance)),
                'max_importance': float(np.max(importance)),
                'min_importance': float(np.min(importance)),
                'n_zero_importance': int(np.sum(importance == 0))
            }
        else:
            return {} 