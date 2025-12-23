# Clean project - Remove generated files for a fresh start
# Removes .pth model files, .md reports, and cache folders
# Preserves MNIST training data in data/ folder

import os
import shutil
from pathlib import Path

def clean_project():
    """Remove generated files and folders"""
    
    print("="*80)
    print("CLEAN PROJECT - Remove Generated Files")
    print("="*80)
    
    workspace = Path(".")
    
    # Define what to clean
    files_to_remove = []
    folders_to_remove = []
    
    # Find .pth files (model checkpoints)
    pth_files = list(workspace.glob("*.pth"))
    files_to_remove.extend(pth_files)
    
    # Find .md files (reports)
    md_files = list(workspace.glob("*.md"))
    # Keep README.md if it exists
    md_files = [f for f in md_files if f.name.lower() != "readme.md"]
    files_to_remove.extend(md_files)
    
    # Find __pycache__ folders
    pycache_folders = list(workspace.glob("**/__pycache__"))
    folders_to_remove.extend(pycache_folders)
    
    # Display what will be removed
    print("\nFiles to be removed:")
    print("-"*80)
    
    if files_to_remove:
        print("\n📄 Files:")
        for f in sorted(files_to_remove):
            size = f.stat().st_size / 1024  # KB
            print(f"  - {f.name} ({size:.1f} KB)")
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
    print("  ✓ data/ folder (MNIST training data)")
    print("  ✓ README.md (if exists)")
    print("  ✓ All Python source files (.py)")
    print("  ✓ All image files (.jpg, .png, etc.)")
    print("="*80)
    
    if not files_to_remove and not folders_to_remove:
        print("\n✓ Project is already clean!")
        return
    
    # Ask for confirmation
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
    print("  1. Run 'python nn_train.py' to train fresh models")
    print("  2. Run 'python test_accuracy.py' to evaluate performance")
    print("  3. Run 'python generate_report.py' to create visual report")
    print("\n💡 To also remove MNIST data, delete the data/ folder manually")
    print("   (This will require re-downloading ~10MB on next training)")

if __name__ == "__main__":
    clean_project()
