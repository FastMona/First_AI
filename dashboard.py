"""
MNIST Digit Recognition System - Main Dashboard
Central console for accessing all project functionalities
Dashboard Menu: This IS the main dashboard - entry point for all options

Menu Structure:
  1. Train with FFN            → calls nn_train_ffn.py
  2. Train with CNN            → calls nn_train_cnn.py
  3. Train with ART            → calls nn_train_art.py
  4. Test Accuracy             → calls test_accuracy.py
  5. Single Image Detection    → calls detect.py
  6. Batch Image Detection     → calls detect_batch.py
  7. Camera Capture            → calls camera.py
  8. Clean Project             → calls clean_project.py
  0. Exit                      → exits dashboard"""

import sys
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Log file for terminal output - session-specific with timestamp
SESSION_TIMESTAMP = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
LOG_FILE = f"dashboard_session_{SESSION_TIMESTAMP}.txt"

class TeeOutput:
    """Class to write to both file and terminal"""
    def __init__(self, file_obj, original_stream):
        self.file_obj = file_obj
        self.original_stream = original_stream
    
    def write(self, text):
        self.file_obj.write(text)
        self.file_obj.flush()
        self.original_stream.write(text)
        self.original_stream.flush()
    
    def flush(self):
        self.file_obj.flush()
        self.original_stream.flush()

def setup_logging():
    """Setup terminal output logging to file"""
    try:
        # Open log file in append mode
        log_file = open(LOG_FILE, 'a', encoding='utf-8')
        
        # Write session header
        log_file.write("\n" + "="*80 + "\n")
        log_file.write(f"Dashboard Session Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write("="*80 + "\n\n")
        log_file.flush()
        
        # Redirect stdout and stderr to both file and console
        sys.stdout = TeeOutput(log_file, sys.__stdout__)
        sys.stderr = TeeOutput(log_file, sys.__stderr__)
        
        return log_file
    except Exception as e:
        logger.warning(f"Warning: Could not setup logging: {e}")
        return None

def close_logging(log_file):
    """Close log file and restore stdout/stderr"""
    try:
        if log_file:
            # Write session footer
            log_file.write("\n" + "="*80 + "\n")
            log_file.write(f"Dashboard Session Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write("="*80 + "\n\n")
            
            # Restore original stdout/stderr
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            
            log_file.close()
    except Exception as e:
        logger.warning(f"Warning: Could not close logging: {e}")

def check_environment():
    """Check if running in the correct PyTorch environment and return environment info"""
    try:
        import torch
        import platform
        import subprocess
        
        # Get Python version
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        
        # Check if CUDA is available (optional but good to know)
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda if cuda_available else None
        
        # Check environment - support both conda and venv
        conda_env = os.environ.get('CONDA_DEFAULT_ENV', None)
        venv_env = os.environ.get('VIRTUAL_ENV', None)
        
        if venv_env:
            env_name = f"venv ({os.path.basename(venv_env)})"
        elif conda_env:
            env_name = conda_env
        else:
            env_name = "base"
        
        # Get CPU information - try multiple methods for Windows
        cpu_name = None
        if platform.system() == 'Windows':
            try:
                # Try WMIC on Windows for detailed CPU info
                result = subprocess.run(
                    ['wmic', 'cpu', 'get', 'name'],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if len(lines) > 1:
                        cpu_name = lines[1].strip()
            except:
                pass
        
        # Fallback to platform.processor()
        if not cpu_name:
            cpu_name = platform.processor()
        
        if not cpu_name or cpu_name == '':
            cpu_name = f"Unknown CPU ({platform.machine()})"
        
        # Return environment info to be displayed in header
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
        
        return {
            'env_name': env_name,
            'python_version': python_version,
            'pytorch_version': torch.__version__,
            'cuda_version': cuda_version,
            'cpu_name': cpu_name,
            'cuda_available': cuda_available,
            'gpu_name': gpu_name
        }
        
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

def print_header(env_info):
    """Print dashboard header with environment information"""
    print("\n" + "="*80)
    print("  MNIST DIGIT RECOGNITION SYSTEM - DASHBOARD".center(80))
    print("="*80)
    
    # Display environment info
    print(f"\n  Environment: {env_info['env_name']}")
    print(f"  Python: {env_info['python_version']} | PyTorch: {env_info['pytorch_version']}", end="")
    if env_info['cuda_version']:
        print(f" | CUDA: {env_info['cuda_version']}")
    else:
        print()
    print(f"  CPU: {env_info['cpu_name']}")
    if env_info['cuda_available']:
        print(f"  GPU: {env_info['gpu_name']}")
        print(f"  ⚡ Compute Device: GPU - {env_info['gpu_name']}")
    else:
        print(f"  💻 Compute Device: CPU")
    print("="*80)

def print_menu():
    """Display main menu options"""
    print("\n┌─ MODEL TRAINING & EVALUATION ─────────────────────────────────────────────┐")
    print("│  1. Train with FFN            - Train simple feedforward network (baseline)│")
    print("│  2. Train with CNN            - Train CNN classifier and autoencoder       │")
    print("│  3. Train with ART            - Train Fuzzy ART classifier and autoencoder │")
    print("│  4. Test Accuracy             - Compare all trained models                 │")
    print("└────────────────────────────────────────────────────────────────────────────┘")
    
    print("\n┌─ IMAGE DETECTION ──────────────────────────────────────────────────────────┐")
    print("│  5. Single Image Detection    - Detect digit in one image (detailed)       │")
    print("│  6. Batch Image Detection     - Process all images in a folder             │")
    print("└────────────────────────────────────────────────────────────────────────────┘")
    
    print("\n┌─ IMAGE CAPTURE & GENERATION ───────────────────────────────────────────────┐")
    print("│  7. Camera Capture            - Capture images from webcam                 │")
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

def run_clean_project(log_file=None):
    """Special handler for clean_project with confirmation"""
    try:
        print("\n" + "─"*80)
        import clean_project
        
        # Close the current log file so it can be deleted
        current_log_file = LOG_FILE
        if log_file:
            close_logging(log_file)
        
        # Run cleanup with interactive prompts for all options
        clean_project.clean_project(interactive=True, current_log_file=current_log_file)
        
        # Don't reopen log file - assume user will exit after cleanup
            
    except ImportError as e:
        print(f"Error importing clean_project: {e}")
    except Exception as e:
        print(f"Error running clean_project: {e}")
    finally:
        print("\n" + "─"*80)
        input("\nPress Enter to return to dashboard...")
    
    return None  # Return None since log file is closed

def run_detect():
    """Special handler for single image detection"""
    try:
        print("\n" + "─"*80)
        import detect
        
        # Prompt for image path
        image_path = input("Enter image filename (e.g., test_images/img_1.jpg): ").strip()
        if image_path:
            detect.main(image_path)
        else:
            print("❌ No image path provided")
            
    except ImportError as e:
        print(f"Error importing detect: {e}")
    except Exception as e:
        print(f"Error running detect: {e}")
    finally:
        print("\n" + "─"*80)
        input("\nPress Enter to return to dashboard...")

def run_detect_batch():
    """Special handler for batch image detection"""
    try:
        print("\n" + "─"*80)
        import detect_batch
        
        # detect_batch.main() will handle model selection internally
        detect_batch.main()
            
    except ImportError as e:
        print(f"Error importing detect_batch: {e}")
    except Exception as e:
        print(f"Error running detect_batch: {e}")
    finally:
        print("\n" + "─"*80)
        input("\nPress Enter to return to dashboard...")

def main():
    """Main dashboard loop"""
    # Setup logging
    log_file = setup_logging()
    
    # Check environment at startup
    env_info = check_environment()
    
    try:
        while True:
            clear_screen()
            print_header(env_info)
            print_menu()
            
            choice = input("\n➤ Select option (0-8): ").strip()
            
            if choice == '1':
                run_module("nn_train_ffn")
            elif choice == '2':
                run_module("nn_train_cnn")
            elif choice == '3':
                run_module("nn_train_art")
            elif choice == '4':
                run_module("test_accuracy")
            elif choice == '5':
                run_detect()
            elif choice == '6':
                run_detect_batch()
            elif choice == '7':
                run_module("camera")
            elif choice == '8':
                log_file = run_clean_project(log_file)
            elif choice == '0':
                close_logging(log_file)
                clear_screen()
                print("\n" + "="*80)
                print("  Thank you for using MNIST Digit Recognition System!".center(80))
                print("="*80 + "\n")
                sys.exit(0)
            else:
                print("\n❌ Invalid option. Please select 0-8.")
                input("Press Enter to continue...")
    except KeyboardInterrupt:
        close_logging(log_file)
        clear_screen()
        print("\n" + "="*80)
        print("  Goodbye!".center(80))
        print("="*80 + "\n")

if __name__ == "__main__":
    log_file = None
    try:
        main()
    except KeyboardInterrupt:
        close_logging(log_file)
        clear_screen()
        print("\n\n" + "="*80)
        print("  Dashboard closed.".center(80))
        print("="*80 + "\n")
        sys.exit(0)
    except Exception as e:
        close_logging(log_file)
        print(f"\nError: {e}")
        sys.exit(1)
