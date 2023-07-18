import os
from subprocess import call

if __name__ == "__main__":
    
    print("STARTING PROJECT TEMPLATE UPDATE PROCESS")
    print("========================================")

    # download latest update.py into utils folder
    update_web = "https://raw.githubusercontent.com/neurorishika/rpy-template/main/utils/update.py"
    print("Downloading latest update.py into utils folder...")
    os.system("curl {} -o utils/update.py.tmp".format(update_web))
    print("Download complete.")
    #check if update.py.tmp is different from update.py
    if os.system("diff utils/update.py.tmp utils/update.py") != 0:
        # replace update.py with update.py.tmp
        os.system("mv utils/update.py.tmp utils/update.py")
        # exit and ask user to run update.py again
        print("An update fpr update.py was found and replaced. Please run 'poetry run python utils/update.py' again.")
        exit(-1)
    else:
        # delete update.py.tmp
        os.system("rm utils/update.py.tmp")
        print("No updates for update.py found.")

    # get package name from pyproject.toml
    with open("pyproject.toml", "r") as f:
        lines = f.readlines()
    for line in lines:
        if "name" in line:
            package_name = line.split("=")[1].strip().replace('"', '')
    print("Package name: {}".format(package_name))

    # download latest rdp_client.py into package folder
    rdp_client_web = "https://raw.githubusercontent.com/neurorishika/rdp-standard/main/rdp_client.py"

    print("Downloading latest rdp_client.py into package folder...")
    os.system("curl {} -o {}/rdp_client.py".format(rdp_client_web, package_name))
    print("Download complete.")

    # download latest build.py and quickstart.py into utils folder
    build_web = "https://raw.githubusercontent.com/neurorishika/rpy-template/main/utils/build.py"
    quickstart_web = "https://raw.githubusercontent.com/neurorishika/rpy-template/main/utils/quickstart.py"

    print("Downloading latest build.py and quickstart.py into utils folder...")
    os.system("curl {} -o utils/build.py".format(build_web))
    os.system("curl {} -o utils/quickstart.py".format(quickstart_web))
    print("Download complete.")

    # download latest pyproject.toml into package folder
    pyproject_web = "https://raw.githubusercontent.com/neurorishika/rpy-template/main/pyproject.toml"

    print("Downloading latest pyproject.toml into package folder...")
    os.system("curl {} -o pyproject.toml.tmp".format(pyproject_web))
    print("Download complete.")

    # go through pyproject.toml.tmp and:
    # 1. find all dependencies that are in the template but not in the current package
    # 2. add those dependencies to the current package

    with open("pyproject.toml.tmp", "r") as f:
        lines_tmp = f.readlines()
    
    dependencies_tmp = []
    full_dependencies_tmp = []
    start = False
    for line in lines_tmp:
        if line == "[tool.poetry.dependencies]\n":
            start = True
            continue
        if start:
            if line == "\n":
                break
            dependencies_tmp.append(line.split("=")[0].strip())
            full_dependencies_tmp.append(line)
    
    with open("pyproject.toml", "r") as f:
        lines = f.readlines()

    dependencies = []
    full_dependencies = []
    start = False
    for line in lines:
        if line == "[tool.poetry.dependencies]\n":
            start = True
            continue
        if start:
            if line == "\n":
                break
            dependencies.append(line.split("=")[0].strip())
            full_dependencies.append(line)

    full_dependencies_to_add = []
    for i,dependency in enumerate(dependencies_tmp):
        if dependency not in dependencies:
            full_dependencies_to_add.append(full_dependencies_tmp[i])
    
    start = False
    with open("pyproject.toml", "w") as f:
        for line in lines:
            if line == "[tool.poetry.dependencies]\n":
                start = True 
            if start:
                if line == "\n":
                    for dependency in full_dependencies_to_add:
                        f.write(dependency)
                    f.write(line)
                    start = False
                else:
                    f.write(line)
            else:
                f.write(line)

    # remove pyproject.toml.tmp
    os.system("rm pyproject.toml.tmp")

    # run poetry lock
    print("Running poetry lock...")
    call("poetry lock", shell=True)
    print("Poetry lock complete.")

    print("PROJECT TEMPLATE UPDATE COMPLETE")