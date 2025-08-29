from setuptools import setup, find_packages

setup(
    name="changepoint-toolkit",
    version="0.1.0",
    author="Change-Point Detection Contributors",
    description="Comprehensive toolkit for change-point detection in time series",
    packages=find_packages(),
    py_modules=["kcp_rff"],
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.5.0",
        "pandas>=1.5.0",
    ],
    entry_points={
        "console_scripts": [
            "bocpd-cli=bocpd.bocpd_cli:main",
            "within-period-cli=within_period.cli:main",
            "cpd-cli=toolkit.cpd_cli:main",
        ],
    },
)
