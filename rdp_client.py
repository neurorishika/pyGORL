from cryptography.fernet import Fernet
import zipfile
import os

def unlock_and_unzip_file(data2unzip, key_dir='key.key'):
    # check if key exists
    try:
        with open(key_dir, "rb") as key_file:
            key = key_file.read()
    except:
        print("Key not found, please generate a key using generate_key(), provide an existing key_dir, or locate the key.")
    fernet = Fernet(key)
    with open(data2unzip, "rb") as file:
        encrypted_data = file.read()
    decrypted_data = fernet.decrypt(encrypted_data)
    data2unzip = data2unzip.split(".")[0]+"_unlocked."+data2unzip.split(".")[1]
    with open(data2unzip, "wb") as file:
        file.write(decrypted_data)
    # unzip all files and folders
    with zipfile.ZipFile(data2unzip, 'r') as zip_ref:
        zip_ref.extractall(data2unzip.split("_unlocked.")[0])
    # delete zip file
    os.remove(data2unzip)

def zip_and_lock_folder(data2zip,key_dir='key.key'):
    # check if key exists
    try:
        with open(key_dir, "rb") as key_file:
            key = key_file.read()
    except:
        print("Key not found, please generate a key using generate_key(), provide an existing key_dir, or locate the key.")
    fernet = Fernet(key)
    # make sure data2zip is a folder
    assert os.path.isdir(data2zip), "data2zip is not a folder"
    # zip folder while preserving directory structure
    fullzip = zipfile.ZipFile(f'{data2zip}.zip', 'w') # create zipfile object
    rootlen = len(data2zip) + 1 # get number of characters to remove from each file path
    for folder, subfolders, files in os.walk(f'{data2zip}'): # walk through folders
        for file in files:
            fn = os.path.join(folder, file)
            fullzip.write(fn, fn[rootlen:], compress_type = zipfile.ZIP_DEFLATED)
        for subfolder in subfolders:
            fn = os.path.join(folder, subfolder)
    fullzip.close() # close zipfile object
    # encrypt zip file
    with open(f'{data2zip}.zip', 'rb') as file:
        original = file.read()
    encrypted = fernet.encrypt(original)
    with open(f'{data2zip}.ezip', 'wb') as encrypted_file:
        encrypted_file.write(encrypted)
    # delete zip file
    os.remove(f'{data2zip}.zip')
    