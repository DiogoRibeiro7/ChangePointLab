from setuptools import setup, find_packages

setup(
    name="changepoint-lab",
    version="0.1.0",
    author="Diogo Ribeiro",
    author_email="dfr@esmad.ipp.pt",
    description="ChangePointLab: a unified Python toolkit for changepoint detection in time series.",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.21.0",
    ],
    entry_points={
        "console_scripts": [
            "bocpd-cli=changepoint_lab.cli.bocpd_cli:main",
            "within-period-cli=changepoint_lab.algorithms.bayesian.within_period.cli:main",
            "cpd-cli=toolkit.cpd_cli:main",
        ],
    },
)
