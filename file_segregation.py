import os
import shutil

def organize_files(src_dir):
    # Get a list of all files in the source directory
    files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]

    for file in files:
        # Find the first occurrence of a digit in the filename
        index = next((i for i, c in enumerate(file) if c.isdigit()), None)

        # Check if a digit was found in the filename
        if index is not None:
            # Create the destination directory name by removing one character before the first digit
            dst_dir = file[:index-1]
            dst_dir = (src_dir + "/" +dst_dir)

            # Create the destination directory if it doesn't exist
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)

            # Move the file to the destination directory
            shutil.move(os.path.join(src_dir, file), os.path.join(dst_dir, file))
        else:
            print(f"No digit found in filename: {file}")

# Example usage
root = '/Users/subh1461/Desktop/Gen AI Roadshow/CodeWhisperer/data/'
organize_files(root+'val')
organize_files(root+'train')