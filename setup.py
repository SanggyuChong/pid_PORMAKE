from setuptools import setup, find_packages

# Get dependencies from requirements.txt file.
with open("requirements.txt", "r") as f:
    install_requires = f.readlines()

setup(
    name="pormake",
    version="0.0.2",
    description="Construction of nanoporous materials"
                " from topology and building blocks.",
    install_requires=install_requires,
    author="Sangwon Lee",
    author_email="integratedmailsystem@gmail.com",
    packages=[
        "pormake",
    ],
    package_data={
        "pormake": ["database/**/*"],
    },
    python_requires=">=3.7",
    zip_safe=False
)

# PID: make modifications to the user's ASE package to work with secondary connection points 'Xx'
import ase, shutil
data_init = ase.__file__[:-11] + "data/__init__.py"
shutil.copy("ase_data_init_mod.py", data_init)
