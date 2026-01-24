"""Clean project - Remove generated files for a fresh start.

Removes .pth model files, .md reports, and cache folders while preserving
MNIST training data in training_data/ folder.

Dashboard Menu: Called by Option 8 - "Clean Project"
"""

import os
import shutil
from pathlib import Path

def clean_project(interactive=True, current_log_file=None):
    """Remove generated files and folders
    
    Args:
        interactive (bool): If True, ask for confirmation. If False, proceed automatically.
        current_log_file (str): Name of the current dashboard log file to include in cleanup
    """
    
    print("="*80)
    print("CLEAN PROJECT - Remove Generated Files")
    print("="*80)
    
    workspace = Path(".")
    
    # Define what to clean
    files_to_remove = []
    folders_to_remove = []
    
    # Find .pth files (model checkpoints - includes both CNN and ART models)
    pth_files = list(workspace.glob("*.pth"))
    files_to_remove.extend(pth_files)
    
    # Explicitly check for all model files
    model_files_found = []
    if Path('model_state.pth').exists():
        model_files_found.append('model_state.pth (CNN)')
    if Path('model_state_art.pth').exists():
        model_files_found.append('model_state_art.pth (ART)')
    if Path('model_state_ffn.pth').exists():
        model_files_found.append('model_state_ffn.pth (FFN)')
    if Path('autoencoder.pth').exists():
        model_files_found.append('autoencoder.pth')
    if Path('ood_params.pth').exists():
        model_files_found.append('ood_params.pth')
    
    # Find .md files (reports)
    md_files = list(workspace.glob("*.md"))
    # Keep README.md if it exists
    md_files = [f for f in md_files if f.name.lower() != "readme.md"]
    files_to_remove.extend(md_files)
    
    # Find session log files (dashboard_session_*.txt) - handled separately
    session_logs = list(workspace.glob("dashboard_session_*.txt"))
    
    # Add current log file if provided (in case it's not found by glob because it's open)
    if current_log_file:
        current_log_path = Path(current_log_file)
        if current_log_path.exists() and current_log_path not in session_logs:
            session_logs.append(current_log_path)
    
    # Find .png files (generated visualizations/plots)
    png_files = list(workspace.glob("*.png"))
    files_to_remove.extend(png_files)
    
    # Find __pycache__ folders
    pycache_folders = list(workspace.glob("**/__pycache__"))
    folders_to_remove.extend(pycache_folders)
    
    # Display what will be removed
    print("\nFiles to be removed:")
    print("-"*80)
    
    if files_to_remove:
        print("\n📄 Files:")
        if model_files_found:
            print("  Model files:")
            for model_name in model_files_found:
                filename = model_name.split(' (')[0]
                if Path(filename).exists():
                    size = Path(filename).stat().st_size / 1024
                    print(f"    - {model_name} ({size:.1f} KB)")
        print("  Other files:")
        for f in sorted(files_to_remove):
            if f.name not in ['model_state.pth', 'model_state_art.pth', 'model_state_ffn.pth', 'autoencoder.pth', 'ood_params.pth']:
                size = f.stat().st_size / 1024  # KB
                print(f"    - {f.name} ({size:.1f} KB)")
    else:
        print("  No files to remove")
    
    if folders_to_remove:
        print("\n📁 Folders:")
        for folder in sorted(folders_to_remove):
            # Calculate folder size
            total_size = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())
            total_size_kb = total_size / 1024
            print(f"  - {folder} ({total_size_kb:.1f} KB)")
    else:
        print("  No folders to remove")
    
    # Display session logs separately
    if session_logs:
        print("\n📋 Session Logs (optional cleanup):")
        total_log_size = 0
        for log in sorted(session_logs):
            size = log.stat().st_size / 1024
            total_log_size += size
            print(f"  - {log.name} ({size:.1f} KB)")
        print(f"  Total: {len(session_logs)} files ({total_log_size:.1f} KB)")
    
    print("\n" + "="*80)
    print("PRESERVED (NOT deleted):")
    print("-"*80)
    print("  ✓ training_data/ folder (MNIST training data)")
    print("  ✓ test_images/ folder (if exists)")
    print("  ✓ README.md (if exists)")
    print("  ✓ All Python source files (.py)")
    if session_logs:
        print("  ⚠️  dashboard_session_*.txt (Will ask separately)")
    print("="*80)
    
    # Initialize variables for optional cleanup
    remove_session_logs = False
    remove_mnist = False
    mnist_folders = []
    
    # Check for MNIST data folders
    if Path('training_data/MNIST').exists():
        mnist_folders.append(Path('training_data/MNIST'))
    if Path('data/MNIST').exists():
        mnist_folders.append(Path('data/MNIST'))
    
    # Check if there's anything to clean
    if not files_to_remove and not folders_to_remove and not session_logs and not mnist_folders:
        print("\n✓ Project is already clean!")
        return
    
    # Ask all questions first (in interactive mode)
    proceed_with_main_cleanup = True
    if interactive:
        # Ask about main cleanup if there are files/folders to remove
        if files_to_remove or folders_to_remove:
            print("\n⚠️  WARNING: This action cannot be undone!")
            response = input("Do you want to proceed with cleanup? (yes/no): ").strip().lower()
            proceed_with_main_cleanup = response in ['yes', 'y']
        
        # Ask about session logs
        if session_logs:
            print("\n" + "-"*80)
            print("📋 Session Log Files:")
            print(f"   Found {len(session_logs)} session log file(s)")
            print("   These files contain terminal output history for demos/documentation")
            response_logs = input("Do you want to delete session logs? (yes/no): ").strip().lower()
            remove_session_logs = response_logs in ['yes', 'y']
        
        # Ask about MNIST data
        if mnist_folders:
            print("\n" + "-"*80)
            print("💾 MNIST Data:")
            total_mnist_size = 0
            for folder in mnist_folders:
                size = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())
                total_mnist_size += size
                print(f"   - {folder} ({size / (1024*1024):.1f} MB)")
            print(f"   Total: {total_mnist_size / (1024*1024):.1f} MB")
            print("   ⚠️  This will require re-downloading on next training (~10MB)")
            response_mnist = input("Do you want to delete MNIST data? (yes/no): ").strip().lower()
            remove_mnist = response_mnist in ['yes', 'y']
        
        # If user said no to everything, exit
        if not proceed_with_main_cleanup and not remove_session_logs and not remove_mnist:
            print("\n❌ Cleanup cancelled")
            return
    
    # Perform cleanup
    print("\n" + "="*80)
    print("CLEANING...")
    print("-"*80)
    
    removed_count = 0
    
    # Remove files (only if user agreed to main cleanup)
    if files_to_remove and proceed_with_main_cleanup:
        for f in files_to_remove:
            try:
                f.unlink()
                print(f"  ✓ Removed: {f.name}")
                removed_count += 1
            except Exception as e:
                print(f"  ✗ Failed to remove {f.name}: {e}")
    
    # Remove folders (only if user agreed to main cleanup)
    if folders_to_remove and proceed_with_main_cleanup:
        for folder in folders_to_remove:
            try:
                shutil.rmtree(folder)
                print(f"  ✓ Removed: {folder}")
                removed_count += 1
            except Exception as e:
                print(f"  ✗ Failed to remove {folder}: {e}")
    
    # Handle session logs based on user choice
    if session_logs and remove_session_logs:
        if removed_count > 0:
            print()
        print("Removing session logs...")
        for log in session_logs:
            try:
                log.unlink()
                print(f"  ✓ Removed: {log.name}")
                removed_count += 1
            except Exception as e:
                print(f"  ✗ Failed to remove {log.name}: {e}")
    elif session_logs and not remove_session_logs:
        if removed_count > 0:
            print()
        print(f"  ✓ Session logs preserved ({len(session_logs)} files)")
    
    # Handle MNIST data based on user choice
    if mnist_folders and remove_mnist:
        if removed_count > 0 or session_logs:
            print()
        print("Removing MNIST data...")
        for folder in mnist_folders:
            try:
                shutil.rmtree(folder)
                print(f"  ✓ Removed: {folder}")
                removed_count += 1
            except Exception as e:
                print(f"  ✗ Failed to remove {folder}: {e}")
    elif mnist_folders and not remove_mnist:
        if removed_count > 0 or session_logs:
            print()
        print(f"  ✓ MNIST data preserved ({len(mnist_folders)} folders)")
    
    print("-"*80)
    if removed_count > 0:
        print(f"\n✓ Cleanup complete! Removed {removed_count} items")
    else:
        print(f"\n✓ No items were removed (all optional items were preserved)")
    print("="*80)
    
    if remove_mnist:
        print("\n💡 MNIST data was deleted - it will be re-downloaded (~10MB) on next training")

def main():
    """Main entry point for direct execution"""
    clean_project(interactive=True)

if __name__ == "__main__":
    main()
