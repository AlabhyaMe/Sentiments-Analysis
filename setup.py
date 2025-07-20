from setuptools import setup, find_packages
import os

here = os.path.abspath(os.path.dirname(__file__))

# Read README
with open(os.path.join(here, 'README.md'), 'r', encoding='utf-8') as f:
    long_description = f.read()

# Handle requirements
try:
    with open(os.path.join(here, 'requirements.txt'), 'r', encoding='utf-8') as f:
        install_requires = [line.strip() for line in f if line.strip() and not line.startswith('#')]
except FileNotFoundError:
    install_requires = [
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scikit-learn>=1.0.0'
    ]

setup(
    name='quick_sentiments',
    version='0.1.0',
    author='Alabhya Dahal',
    author_email='alabhya.dahal@gmail.com',
    description='Sentiment Analysis pipeline',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/AlabhyaMe/Sentiments-Analysis.git',
    packages=find_packages(),
    package_data={
        'quick_sentiments': ['*.json', '*.pkl'],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    install_requires=install_requires,
    extras_require={
        'dev': [
            'pytest>=6.0',
            'black>=22.0',
        ],
    },
)