from setuptools import find_packages, setup

setup(
    name='ml-project-template',
    packages=find_packages(),
    version='0.1.0',
    description='Template para projetos de Machine Learning',
    author='Seu Nome',
    license='MIT',
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'jupyter',
        'python-dotenv',
        'pyyaml',
        'click',
    ],
    python_requires='>=3.8',
)
