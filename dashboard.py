"""
MNIST Digit Recognition System - Main Dashboard
Central console for accessing all project functionalities
"""

import sys
import os

def check_environment():
    """Check if running in the correct PyTorch environment"""
    try:
        import torch
        # Check if CUDA is available (optional but good to know)
        cuda_available = torch.cuda.is_available()
        
        # Check conda environment name
        conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'Unknown')
        
        # Verify we're in pytorch environment
        if conda_env != 'pytorch':
            print("\n" + "="*80)
            print("  ⚠️  WARNING: NOT IN PYTORCH ENVIRONMENT".center(80))
            print("="*80)
            print(f"\nCurrent environment: {conda_env}")
            print("Expected environment: pytorch")
            print("\nPlease activate the PyTorch environment:")
            print("  conda activate pytorch")
            print("\nThen run the dashboard again:")
            print("  python dashboard.py")
            print("="*80 + "\n")
            sys.exit(1)
        
        # Show environment info
        print(f"\n✓ Environment: {conda_env}")
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {cuda_available}")
        if cuda_available:
            print(f"✓ CUDA device: {torch.cuda.get_device_name(0)}")
        
    except ImportError:
        print("\n" + "="*80)
        print("  ⚠️  ERROR: PyTorch NOT INSTALLED".center(80))
        print("="*80)
        print("\nPyTorch is not installed in this environment.")
        print("\nPlease install the environment:")
        print("  conda env create -f environment.yml")
        print("  conda activate pytorch")
        print("="*80 + "\n")
        sys.exit(1)

def clear_screen():
    """Clear the console screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print dashboard header"""
    print("\n" + "="*80)
    print("  MNIST DIGIT RECOGNITION SYSTEM - DASHBOARD".center(80))
    print("="*80)

def print_menu():
    """Display main menu options"""
    print("\n┌─ MODEL TRAINING & EVALUATION ─────────────────────────────────────────────┐")
    print("│  1. Train Models              - Train CNN classifier and autoencoder       │")
    print("│  2. Test Accuracy             - Test model accuracy on labeled images      │")
    print("│  3. Generate Report           - Create markdown report with visualizations │")
    print("└────────────────────────────────────────────────────────────────────────────┘")
    
    print("\n┌─ IMAGE DETECTION ──────────────────────────────────────────────────────────┐")
    print("│  4. Single Image Detection    - Detect digit in one image (detailed)       │")
    print("│  5. Batch Image Detection     - Process all images in a folder             │")
    print("└────────────────────────────────────────────────────────────────────────────┘")
    
    print("\n┌─ IMAGE CAPTURE & GENERATION ───────────────────────────────────────────────┐")
    print("│  6. Camera Capture            - Capture images from webcam                 │")
    print("└────────────────────────────────────────────────────────────────────────────┘")
    
    print("\n┌─ VISUALIZATION & ANALYSIS ─────────────────────────────────────────────────┐")
    print("│  7. Visualize Conditional AE  - Show class-conditional autoencoder         │")
    print("└────────────────────────────────────────────────────────────────────────────┘")
    
    print("\n┌─ UTILITIES ────────────────────────────────────────────────────────────────┐")
    print("│  8. Clean Project             - Clean up temporary files and cache         │")
    print("│  0. Exit                      - Close dashboard                            │")
    print("└────────────────────────────────────────────────────────────────────────────┘")

def run_module(module_name, function_name="main"):
    """Import and run a module's main function"""
    try:
        print("\n" + "─"*80)
        module = __import__(module_name)
        if hasattr(module, function_name):
            getattr(module, function_name)()
        else:
            print(f"Error: Module '{module_name}' has no function '{function_name}'")
    except ImportError as e:
        print(f"Error importing {module_name}: {e}")
    except Exception as e:
        print(f"Error running {module_name}: {e}")
    finally:
        print("\n" + "─"*80)
        input("\nPress Enter to return to dashboard...")

def main():
    """Main dashboard loop"""
    # Check environment at startup
    check_environment()
    
    while True:
        clear_screen()
        print_header()
        print_menu()
        
        choice = input("\n➤ Select option (0-8): ").strip()
        
        if choice == '1':
            run_module("nn_train")
        elif choice == '2':
            run_module("test_accuracy")
        elif choice == '3':
            run_module("generate_report")
        elif choice == '4':
            run_module("detect")
        elif choice == '5':
            run_module("detect_batch")
        elif choice == '6':
            run_module("camera")
        elif choice == '7':
            run_module("visualize_conditional_ae")
        elif choice == '8':
            run_module("clean_project")
        elif choice == '0':
            clear_screen()
            print("\n" + "="*80)
            print("  Thank you for using MNIST Digit Recognition System!".center(80))
            print("="*80 + "\n")
            sys.exit(0)
        else:
            print("\n❌ Invalid option. Please select 0-8.")
            input("Press Enter to continue...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        clear_screen()
        print("\n\n" + "="*80)
        print("  Dashboard closed.".center(80))
        print("="*80 + "\n")
        sys.exit(0)
