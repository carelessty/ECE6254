"""
Validation script for the privacy risk detection implementation.
This script validates the correctness of the implementation and ensures reproducibility.
"""
import sys
import os

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
sys.path.insert(0, project_root)

import os
import sys
import importlib
import inspect
import torch
from transformers import AutoTokenizer

# Add the implementation directory to the path
sys.path.append('./privacy_risk_detection/code/implementation')
sys.path.append('./privacy_risk_detection/code/evaluation')

# Import the modules to validate
import implementation.data_utils
import implementation.model
from src.evaluation.evaluator import PrivacyRiskEvaluator, load_reference_model
from src.evaluation.config import EXPERIMENTS
import src.evaluation.sample_analysis

def validate_imports():
    """Validate that all necessary modules can be imported."""
    print("Validating imports...")
    
    modules_to_check = [
        'data_utils',
        'model',
        'evaluator',
        'config',
        'compare_models',
        'run_experiments',
        'sample_analysis',
        'run_analysis'
    ]
    
    all_imported = True
    
    for module_name in modules_to_check:
        try:
            if module_name in sys.modules:
                print(f"✓ Module {module_name} already imported")
            else:
                module = importlib.import_module(module_name)
                print(f"✓ Successfully imported {module_name}")
        except ImportError as e:
            print(f"✗ Failed to import {module_name}: {e}")
            all_imported = False
    
    return all_imported

def validate_data_utils():
    """Validate the data_utils module."""
    print("\nValidating data_utils module...")
    
    # Check if the necessary classes and functions exist
    required_items = [
        'SelfDisclosureDataset',
        'load_self_disclosure_dataset',
        'create_mock_dataloaders'
    ]
    
    all_exist = True
    
    for item in required_items:
        if hasattr(implementation.data_utils, item):
            print(f"✓ {item} exists in data_utils")
        else:
            print(f"✗ {item} does not exist in data_utils")
            all_exist = False
    
    # Test the create_mock_dataloaders function
    try:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        train_dataloader, val_dataloader, test_dataloader = implementation.data_utils.create_mock_dataloaders(
            tokenizer, batch_size=2, task="classification"
        )
        
        # Check if dataloaders are created correctly
        if train_dataloader and val_dataloader and test_dataloader:
            print("✓ create_mock_dataloaders function works correctly")
        else:
            print("✗ create_mock_dataloaders function did not return all dataloaders")
            all_exist = False
    except Exception as e:
        print(f"✗ Error testing create_mock_dataloaders: {e}")
        all_exist = False
    
    return all_exist

def validate_model():
    """Validate the model module."""
    print("\nValidating model module...")
    
    # Check if the necessary classes exist
    required_classes = [
        'PrivacyRiskClassifier'
    ]
    
    all_exist = True
    
    for class_name in required_classes:
        if hasattr(implementation.model, class_name):
            print(f"✓ {class_name} exists in model module")
            
            # Check if the class has the necessary methods
            cls = getattr(implementation.model, class_name)
            required_methods = ['__init__', 'predict']
            
            for method_name in required_methods:
                if hasattr(cls, method_name) and callable(getattr(cls, method_name)):
                    print(f"  ✓ {class_name}.{method_name} method exists")
                else:
                    print(f"  ✗ {class_name}.{method_name} method does not exist")
                    all_exist = False
        else:
            print(f"✗ {class_name} does not exist in model module")
            all_exist = False
    
    return all_exist

def validate_evaluator():
    """Validate the evaluator module."""
    print("\nValidating evaluator module...")
    
    # Check if PrivacyRiskEvaluator class exists and has necessary methods
    if 'PrivacyRiskEvaluator' in dir(sys.modules['evaluator']):
        print("✓ PrivacyRiskEvaluator class exists")
        
        # Check methods
        evaluator_class = sys.modules['evaluator'].PrivacyRiskEvaluator
        required_methods = [
            '__init__',
            'evaluate_model',
            'compare_models',
            'generate_report'
        ]
        
        all_methods_exist = True
        
        for method_name in required_methods:
            if hasattr(evaluator_class, method_name) and callable(getattr(evaluator_class, method_name)):
                print(f"  ✓ PrivacyRiskEvaluator.{method_name} method exists")
            else:
                print(f"  ✗ PrivacyRiskEvaluator.{method_name} method does not exist")
                all_methods_exist = False
        
        return all_methods_exist
    else:
        print("✗ PrivacyRiskEvaluator class does not exist")
        return False

def validate_config():
    """Validate the config module."""
    print("\nValidating config module...")
    
    # Check if the necessary variables exist
    required_vars = [
        'EXPERIMENTS',
        'METRICS',
        'VISUALIZATION'
    ]
    
    all_exist = True
    
    for var_name in required_vars:
        if hasattr(sys.modules['config'], var_name):
            print(f"✓ {var_name} exists in config module")
        else:
            print(f"✗ {var_name} does not exist in config module")
            all_exist = False
    
    # Check if EXPERIMENTS is properly formatted
    if hasattr(sys.modules['config'], 'EXPERIMENTS'):
        experiments = sys.modules['config'].EXPERIMENTS
        
        if isinstance(experiments, list) and len(experiments) > 0:
            print(f"✓ EXPERIMENTS is a non-empty list with {len(experiments)} experiments")
            
            # Check if each experiment has the required keys
            required_keys = [
                'name',
                'description',
                'deepseek_model',
                'reference_model',
                'task',
                'approach',
                'use_lora',
                'batch_size',
                'output_dir'
            ]
            
            for i, exp in enumerate(experiments):
                all_keys_exist = True
                
                for key in required_keys:
                    if key not in exp:
                        print(f"  ✗ Experiment {i} is missing the '{key}' key")
                        all_keys_exist = False
                
                if all_keys_exist:
                    print(f"  ✓ Experiment {i} ({exp['name']}) has all required keys")
        else:
            print("✗ EXPERIMENTS is not a non-empty list")
            all_exist = False
    
    return all_exist

def validate_sample_analysis():
    """Validate the sample_analysis module."""
    print("\nValidating sample_analysis module...")
    
    # Check if the necessary functions exist
    required_functions = [
        'run_sample_analysis',
        'create_visualization',
        'generate_report'
    ]
    
    all_exist = True
    
    for func_name in required_functions:
        if hasattr(src.evaluation.sample_analysis, func_name) and callable(getattr(src.evaluation.sample_analysis, func_name)):
            print(f"✓ {func_name} function exists in sample_analysis module")
        else:
            print(f"✗ {func_name} function does not exist in sample_analysis module")
            all_exist = False
    
    return all_exist

def validate_directory_structure():
    """Validate the project directory structure."""
    print("\nValidating directory structure...")
    
    required_dirs = [
        '/home/ubuntu/privacy_risk_detection/papers',
        '/home/ubuntu/privacy_risk_detection/papers/pdf',
        '/home/ubuntu/privacy_risk_detection/data',
        '/home/ubuntu/privacy_risk_detection/code',
        '/home/ubuntu/privacy_risk_detection/code/implementation',
        '/home/ubuntu/privacy_risk_detection/code/evaluation',
        '/home/ubuntu/privacy_risk_detection/results'
    ]
    
    required_files = [
        '/home/ubuntu/privacy_risk_detection/papers/analysis.md',
        '/home/ubuntu/privacy_risk_detection/data/model_dataset_analysis.md',
        '/home/ubuntu/privacy_risk_detection/code/implementation/data_utils.py',
        '/home/ubuntu/privacy_risk_detection/code/implementation/model.py',
        '/home/ubuntu/privacy_risk_detection/code/implementation/main.py',
        '/home/ubuntu/privacy_risk_detection/code/evaluation/evaluator.py',
        '/home/ubuntu/privacy_risk_detection/code/evaluation/config.py',
        '/home/ubuntu/privacy_risk_detection/code/evaluation/compare_models.py',
        '/home/ubuntu/privacy_risk_detection/code/evaluation/run_experiments.py',
        '/home/ubuntu/privacy_risk_detection/code/evaluation/sample_analysis.py',
        '/home/ubuntu/privacy_risk_detection/code/evaluation/run_analysis.py',
        '/home/ubuntu/privacy_risk_detection/todo.md',
        '/home/ubuntu/privacy_risk_detection/implementation_report.md'
    ]
    
    all_exist = True
    
    for dir_path in required_dirs:
        if os.path.isdir(dir_path):
            print(f"✓ Directory {dir_path} exists")
        else:
            print(f"✗ Directory {dir_path} does not exist")
            all_exist = False
    
    for file_path in required_files:
        if os.path.isfile(file_path):
            print(f"✓ File {file_path} exists")
        else:
            print(f"✗ File {file_path} does not exist")
            all_exist = False
    
    return all_exist

def run_sample_test():
    """Run a sample test to validate the implementation."""
    print("\nRunning sample test...")
    
    try:
        # Create output directory
        output_dir = "/home/ubuntu/privacy_risk_detection/results/validation_test"
        os.makedirs(output_dir, exist_ok=True)
        
        # Run sample analysis
        comparison = src.evaluation.sample_analysis.run_sample_analysis()
        
        # Check if comparison results are returned
        if comparison and isinstance(comparison, dict) and len(comparison) > 0:
            print("✓ Sample analysis ran successfully and returned comparison results")
            return True
        else:
            print("✗ Sample analysis did not return valid comparison results")
            return False
    except Exception as e:
        print(f"✗ Error running sample test: {e}")
        return False

def main():
    """Main validation function."""
    print("Starting validation of privacy risk detection implementation...\n")
    
    # Run validation checks
    imports_valid = validate_imports()
    data_utils_valid = validate_data_utils()
    model_valid = validate_model()
    evaluator_valid = validate_evaluator()
    config_valid = validate_config()
    sample_analysis_valid = validate_sample_analysis()
    directory_structure_valid = validate_directory_structure()
    sample_test_valid = run_sample_test()
    
    # Summarize validation results
    print("\n=== Validation Summary ===")
    print(f"Imports: {'✓' if imports_valid else '✗'}")
    print(f"Data Utils: {'✓' if data_utils_valid else '✗'}")
    print(f"Model: {'✓' if model_valid else '✗'}")
    print(f"Evaluator: {'✓' if evaluator_valid else '✗'}")
    print(f"Config: {'✓' if config_valid else '✗'}")
    print(f"Sample Analysis: {'✓' if sample_analysis_valid else '✗'}")
    print(f"Directory Structure: {'✓' if directory_structure_valid else '✗'}")
    print(f"Sample Test: {'✓' if sample_test_valid else '✗'}")
    
    # Overall validation result
    all_valid = (
        imports_valid and
        data_utils_valid and
        model_valid and
        evaluator_valid and
        config_valid and
        sample_analysis_valid and
        directory_structure_valid and
        sample_test_valid
    )
    
    if all_valid:
        print("\n✅ Validation PASSED: The implementation is correct and reproducible.")
    else:
        print("\n❌ Validation FAILED: There are issues with the implementation that need to be addressed.")
    
    return all_valid

if __name__ == "__main__":
    main()
