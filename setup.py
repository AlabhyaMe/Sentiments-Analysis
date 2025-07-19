# setup.py

from setuptools import setup, find_packages

# Read the content of your README.md file
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# Read the content of your requirements.txt file
with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

setup(
    name='quick_sentiments', # The name of your package
    version='0.1.0',             # Initial version
    author='Alabhya Dahal',          # Your name
    author_email='alabhya.dahal@gmail.com', # Your email
    description='An easy-to-use, ready-made Sentiment Analysis pipeline.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/AlabhyaMe/Sentiments-Analysis.git', # Link to your GitHub repo
    packages=find_packages(),    # Automatically finds all packages (folders with __init__.py)
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Text Processing :: Linguistic',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
    ],
    python_requires='>=3.8', # Minimum Python version required
    install_requires=install_requires, # List of dependencies from requirements.txt
)