from setuptools import find_packages, setup

# Required dependencies
ghaia_requirements = [
    "stable-retro",
    "opencv-python",
    "scipy==1.11.4",
    "gymnasium==0.28.1"
]

cl_requirements = [
    "tensorflow==2.11",
    "tensorflow-probability==0.19",
    "wandb",
]

results_processing_requirements = [
    "wandb",
    "matplotlib",
    "seaborn",
    "pandas",
    "jinja2",
]

setup(
    name="GHAIA",
    description="GHAIA: Games Human-AI Alignment Benchmark - Continual RL on Super Mario Bros",
    version='2.0.0',
    url='https://github.com/courtois-neuromod/GHAIA',
    author='Courtois NeuroMod',
    author_email='',
    license='MIT',
    keywords=["human-ai alignment", "continual learning", "super mario bros", "stable-retro", "reinforcement learning", "benchmarking"],
    packages=find_packages(),
    include_package_data=True,
    install_requires=ghaia_requirements,
    extras_require={
        'cl': cl_requirements,
        'results': results_processing_requirements,
    },
)
