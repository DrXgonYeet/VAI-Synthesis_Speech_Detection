import zipfile
import os
import shutil

# CONFIGURATION
BROKEN_MODEL = "voice_auth_model.pth"
REPACKED_MODEL = "voice_auth_model_fixed.pth"

def force_repack():
    print(f"📦 Inspecting '{BROKEN_MODEL}'...")
    
    if not os.path.exists(BROKEN_MODEL):
        print("❌ Error: 'voice_auth_model.pth' not found.")
        return

    # Open the broken zip
    with zipfile.ZipFile(BROKEN_MODEL, 'r') as old_zip:
        # Create a new correct zip
        with zipfile.ZipFile(REPACKED_MODEL, 'w', zipfile.ZIP_STORED) as new_zip:
            
            for item in old_zip.infolist():
                # Read the file data
                file_data = old_zip.read(item.filename)
                
                # ORIGINAL NAME: .data/somefile OR data.pkl
                original_name = item.filename
                
                # 1. Strip any existing "root" slash to be safe
                clean_name = original_name.lstrip("/")
                
                # 2. Check if it already starts with 'archive/'
                if clean_name.startswith("archive/"):
                    # It's already correct, keep it
                    final_name = clean_name
                else:
                    # FORCE it into 'archive/'
                    # This handles 'data.pkl' -> 'archive/data.pkl'
                    # AND '.data/file' -> 'archive/.data/file'
                    final_name = f"archive/{clean_name}"
                
                print(f"  Fixing: {original_name} -> {final_name}")
                new_zip.writestr(final_name, file_data)

    print(f"\n✅ Created valid model: '{REPACKED_MODEL}'")
    
    # Replace the broken file with the fixed one
    if os.path.exists(BROKEN_MODEL):
        os.remove(BROKEN_MODEL)
    os.rename(REPACKED_MODEL, BROKEN_MODEL)
    print("✅ Replaced original file successfully.")
    print("🚀 You can now run 'python train.py'!")

if __name__ == "__main__":
    force_repack()