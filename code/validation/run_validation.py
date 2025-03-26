"""
Script to run the validation and generate a validation report.
"""

import os
import sys
import subprocess
import datetime

def run_validation():
    """Run the validation script and capture the output."""
    print("Running validation script...")
    
    # Create output directory
    output_dir = "/home/ubuntu/privacy_risk_detection/results/validation"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run the validation script
    validation_script = "/home/ubuntu/privacy_risk_detection/code/validation/validate_implementation.py"
    
    try:
        result = subprocess.run(
            [sys.executable, validation_script],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Save the output to a file
        output_file = os.path.join(output_dir, "validation_output.txt")
        with open(output_file, "w") as f:
            f.write(result.stdout)
        
        print(f"Validation output saved to {output_file}")
        
        # Generate validation report
        generate_report(result.stdout, output_dir)
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"Validation failed with error: {e}")
        
        # Save the error output to a file
        error_file = os.path.join(output_dir, "validation_error.txt")
        with open(error_file, "w") as f:
            f.write(e.stdout)
            f.write("\n\nError:\n")
            f.write(e.stderr)
        
        print(f"Validation error saved to {error_file}")
        
        return False

def generate_report(validation_output, output_dir):
    """Generate a validation report based on the validation output."""
    print("Generating validation report...")
    
    # Parse validation results
    validation_results = {}
    
    for line in validation_output.split("\n"):
        if line.startswith("Imports:"):
            validation_results["Imports"] = "✓" in line
        elif line.startswith("Data Utils:"):
            validation_results["Data Utils"] = "✓" in line
        elif line.startswith("Model:"):
            validation_results["Model"] = "✓" in line
        elif line.startswith("Evaluator:"):
            validation_results["Evaluator"] = "✓" in line
        elif line.startswith("Config:"):
            validation_results["Config"] = "✓" in line
        elif line.startswith("Sample Analysis:"):
            validation_results["Sample Analysis"] = "✓" in line
        elif line.startswith("Directory Structure:"):
            validation_results["Directory Structure"] = "✓" in line
        elif line.startswith("Sample Test:"):
            validation_results["Sample Test"] = "✓" in line
    
    # Determine overall validation result
    all_valid = all(validation_results.values())
    
    # Create report
    report = "# Privacy Risk Detection Implementation Validation Report\n\n"
    report += f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    report += "## Validation Summary\n\n"
    
    if all_valid:
        report += "**Overall Result: ✅ PASSED**\n\n"
        report += "The implementation has been validated and is correct and reproducible.\n\n"
    else:
        report += "**Overall Result: ❌ FAILED**\n\n"
        report += "There are issues with the implementation that need to be addressed.\n\n"
    
    report += "### Component Validation Results\n\n"
    report += "| Component | Result |\n"
    report += "|-----------|--------|\n"
    
    for component, result in validation_results.items():
        report += f"| {component} | {'✅ PASSED' if result else '❌ FAILED'} |\n"
    
    report += "\n## Validation Details\n\n"
    report += "The validation process checked the following aspects of the implementation:\n\n"
    report += "1. **Imports**: Verified that all necessary modules can be imported\n"
    report += "2. **Data Utils**: Validated the data processing utilities\n"
    report += "3. **Model**: Checked the model implementation classes and methods\n"
    report += "4. **Evaluator**: Validated the evaluation framework\n"
    report += "5. **Config**: Verified the experiment configurations\n"
    report += "6. **Sample Analysis**: Checked the sample analysis implementation\n"
    report += "7. **Directory Structure**: Validated the project directory structure\n"
    report += "8. **Sample Test**: Ran a sample test to verify functionality\n\n"
    
    report += "## Reproducibility\n\n"
    
    if validation_results.get("Sample Test", False):
        report += "The implementation has been verified to be reproducible. The sample test ran successfully, demonstrating that the code can be executed and produces the expected results.\n\n"
        report += "To reproduce the results:\n\n"
        report += "1. Run the sample analysis script:\n"
        report += "   ```bash\n"
        report += "   cd /home/ubuntu/privacy_risk_detection/code/evaluation\n"
        report += "   python sample_analysis.py\n"
        report += "   ```\n\n"
        report += "2. For the full analysis with mock data:\n"
        report += "   ```bash\n"
        report += "   cd /home/ubuntu/privacy_risk_detection/code/evaluation\n"
        report += "   python run_analysis.py --experiment sentence_classification_few_shot --use_mock_data\n"
        report += "   ```\n\n"
    else:
        report += "The implementation failed the reproducibility test. The sample test did not run successfully, indicating issues with the code execution or result generation.\n\n"
        report += "Please check the validation output for details on the specific issues encountered.\n\n"
    
    report += "## Conclusion\n\n"
    
    if all_valid:
        report += "The implementation has been thoroughly validated and meets all requirements for correctness and reproducibility. It provides a solid foundation for adapting DeepSeek-R1-Distill-Qwen-1.5B for privacy risk detection in user-generated content and comparing its performance with the reference model.\n\n"
    else:
        report += "The implementation has issues that need to be addressed before it can be considered valid and reproducible. Please refer to the validation output for specific details on the issues encountered.\n\n"
    
    # Save report
    report_file = os.path.join(output_dir, "validation_report.md")
    with open(report_file, "w") as f:
        f.write(report)
    
    print(f"Validation report saved to {report_file}")

if __name__ == "__main__":
    run_validation()
