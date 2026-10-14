import os

import h5py
import numpy as np

# Change this to your actual .h5 filename
h5_filename = 'resources/food_test_c101_n1000_r128x128x3.h5'
# The name of the real folder where everything will be saved
output_directory = 'extracted_files'

def unpack_h5(name, obj):
    # Construct the actual folder/file path on your hard drive
    target_path = os.path.join(output_directory, name)
    
    if isinstance(obj, h5py.Group):
        # Recreate the internal folder structure on your computer
        os.makedirs(target_path, exist_ok=True)
        print(f"Created folder: {target_path}")
        
    elif isinstance(obj, h5py.Dataset):
        # Make sure the parent directory exists
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        # Read the raw data
        data = obj[()]
        
        # Save textual or string data as standard .txt files
        if isinstance(data, bytes) or (isinstance(data, np.ndarray) and data.dtype.kind in ['S', 'U']):
            with open(f"{target_path}.txt", 'w', encoding='utf-8') as f:
                if isinstance(data, np.ndarray):
                    f.writelines(f"{item.decode('utf-8') if isinstance(item, bytes) else item}\n" for item in data.flatten())
                else:
                    f.write(data.decode('utf-8'))
            print(f"Extracted text: {target_path}.txt")
            
        # Save numeric arrays as standard binary .npy files to keep exact data structure
        else:
            np.save(target_path, data)
            print(f"Extracted data: {target_path}.npy")

# Open the file and recursively unpack every folder and file inside
with h5py.File(h5_filename, 'r') as f:
    f.visititems(unpack_h5)

print("\n Extraction complete! Check the 'extracted_files' folder.")