from setuptools import setup, find_packages

setup(
    name="advanced-rl-trader",
    version="2.0.0",
    description="Advanced SAC RL Trading Agent with CNN-LSTM-Transformer architecture",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "pyyaml>=6.0",
        "matplotlib>=3.5.0",
        "tensorboard>=2.10.0",
    ],
    extras_require={
        "tune": ["optuna>=3.0.0"],
        "dev":  ["pytest", "black", "isort"],
    },
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "rl-train=scripts.train:main",
            "rl-backtest=scripts.backtest:main",
            "rl-monitor=scripts.monitor:main",
        ],
    },
)
