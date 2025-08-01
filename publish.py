import re
import shutil
import subprocess
import inspect
import importlib
from pathlib import Path


def create_init_file():
    init_file = Path('./src/rgwfuncs/__init__.py')
    init_file.parent.mkdir(parents=True, exist_ok=True)

    init_file_content = "# This file is automatically generated\n"
    init_file_content += "# Dynamically importing functions from modules\n\n"

    src_path = Path('./src')
    rgwfuncs_path = src_path / 'rgwfuncs'

    # Ensure rgwfuncs module can be imported
    import sys
    sys.path.insert(0, str(src_path))

    for module_file in rgwfuncs_path.glob('*.py'):
        module_name = module_file.stem
        if module_name != "__init__":
            mod = importlib.import_module(f'rgwfuncs.{module_name}')
            functions = [name for name, obj in inspect.getmembers(mod, inspect.isfunction)
                         if obj.__module__ == f'rgwfuncs.{module_name}']

            if functions:
                functions_list = ', '.join(functions)
                init_file_content += f"from .{module_name} import {functions_list}\n"

    with init_file.open('w') as f:
        f.write(init_file_content)

    print("Generated __init__.py with content:")
    print(init_file_content)


def increment_version():
    version_file = Path("pyproject.toml")

    # Read the current version from pyproject.toml
    with version_file.open('r') as f:
        content = f.read()

    current_version_match = re.search(r'version = "([0-9]+\.[0-9]+\.[0-9]+)"', content)
    if current_version_match:
        current_version = current_version_match.group(1)

        # Increment the patch version
        version_parts = current_version.split('.')
        version_parts[2] = str(int(version_parts[2]) + 1)
        new_version = '.'.join(version_parts)

        # Update the version in pyproject.toml
        new_content = content.replace(current_version, new_version)
        with version_file.open('w') as f:
            f.write(new_content)

        print(f'Updated version to {new_version}')
        return new_version
    else:
        raise RuntimeError("Version line not found in pyproject.toml")


def rebuild_package():
    subprocess.run(['python3', '-m', 'pip', 'install', '--upgrade', 'build', 'twine'])
    dist_dir = Path('dist')
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    subprocess.run(['python3', '-m', 'build'])


def upload_package():
    subprocess.run(['python3', '-m', 'twine', 'upload', 'dist/*'])


def verify_package(new_version):
    print(f"Execute this command after a minute to verify the new version {new_version}:")
    print("pip3 install --upgrade rgwfuncs")


if __name__ == "__main__":
    print("Creating __init__.py...")
    create_init_file()

    print("Incrementing version...")
    new_version = increment_version()

    print("Rebuilding package...")
    rebuild_package()

    print("Uploading package...")
    upload_package()

    print("Verifying package...")
    verify_package(new_version)
