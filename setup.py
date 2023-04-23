### if we want the project to be considered as a package then we have to put the details in the setup.py
from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'
def get_requirements(file_path:str)->List[str]:
    requirements=[]

    with open(file_path) as file_obj:
        reqirements=file_obj.readlines()
        reqirements = [req.replace("\n", "") for req in requirements]
   
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
   
    return requirements


setup(
    name = "HppRegressionProject",
    version= "0.0.1",
    author="Bratati",
    author_email="xyz@gmail.com",
    install_reuqires=get_requirements("requirements.txt"),
    packages = find_packages()

)