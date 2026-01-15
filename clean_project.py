"""Clean project - Remove generated files for a fresh start.

Removes .pth model files, .md reports, and cache folders while preserving
MNIST training data in training_data/ folder.
"""

import os
import shutil
from pathlib import Path

def clean_project(interactive=True):
    """Remove generated files and folders
    
    Args:
        interactive (bool): If True, ask for confirmation. If False, proceed automatically.
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
    
    print("\n" + "="*80)
    print("PRESERVED (NOT deleted):")
    print("-"*80)
    print("  ✓ training_data/ folder (MNIST training data)")
    print("  ✓ test_images/ folder (if exists)")
    print("  ✓ README.md (if exists)")
    print("  ✓ All Python source files (.py)")
    print("="*80)
    
    if not files_to_remove and not folders_to_remove:
        print("\n✓ Project is already clean!")
        return
    
    # Ask for confirmation (only if interactive mode)
    if interactive:
        print("\n⚠️  WARNING: This action cannot be undone!")
        response = input("Do you want to proceed? (yes/no): ").strip().lower()
        
        if response not in ['yes', 'y']:
            print("\n❌ Cleanup cancelled")
            return
    
    # Perform cleanup
    print("\n" + "="*80)
    print("CLEANING...")
    print("-"*80)
    
    removed_count = 0
    
    # Remove files
    for f in files_to_remove:
        try:
            f.unlink()
            print(f"  ✓ Removed: {f.name}")
            removed_count += 1
        except Exception as e:
            print(f"  ✗ Failed to remove {f.name}: {e}")
    
    # Remove folders
    for folder in folders_to_remove:
        try:
            shutil.rmtree(folder)
            print(f"  ✓ Removed: {folder}")
            removed_count += 1
        except Exception as e:
            print(f"  ✗ Failed to remove {folder}: {e}")
    
    print("-"*80)
    print(f"\n✓ Cleanup complete! Removed {removed_count} items")
    print("="*80)
    
    # Show next steps
    print("\nNEXT STEPS:")
    print("  1. Train models via dashboard:")
    print("     - Option 1: Train with CNN")
    print("     - Option 2: Train with ART")
    print("  2. Run 'python test_accuracy.py' to compare both models")
    print("  3. Run 'python generate_report.py' to create visual report")
    print("\n💡 To also remove MNIST data, delete the training_data/ folder manually")
    print("   (This will require re-downloading ~10MB on next training)")

def main():
    """Main entry point for direct execution"""
    clean_project(interactive=True)

if __name__ == "__main__":
    main()
