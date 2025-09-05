# JOSS Review Checklist

This checklist is for authors to help prepare their submission for JOSS review. Submitted papers that don't meet these guidelines may be rejected without review.

## Repository Structure

The BOCPD package repository includes:

- [x] The software is available in a public repository (GitHub, GitLab, Bitbucket, etc.)
- [x] The software includes an OSI-approved open source license
- [x] The repository includes documentation clearly stating how to install and use the software
- [x] The repository has a README.md file that:
  - [x] Explains the purpose of the software
  - [x] Provides installation instructions
  - [x] Includes example usage
  - [x] Contains appropriate citations for the software
  - [x] Lists dependencies
- [x] The repository includes sufficient API documentation
- [x] The software includes tests with reasonable coverage
- [x] The repository is structured in a way that is conventional for the programming language used
- [x] The software includes continuous integration setup (GitHub Actions, Travis CI, etc.)

## Software Functionality

The BOCPD software:

- [x] Performs the research functions described in the paper
- [x] Has clearly defined functionality with well-defined inputs and outputs
- [x] Is well-structured, employing appropriate data structures and algorithms
- [x] Includes tests that verify correct behavior
- [x] Has documentation that explains how the software works

## JOSS Paper

The paper includes:

- [x] A statement of need that clearly articulates the research purpose of the software
- [x] A summary of the software's functionality and how it addresses the stated need
- [x] A description of how the software compares to other similar packages, highlighting the novel aspects
- [x] Citations to relevant literature
- [x] Example usage of the key functionality
- [x] Clear figures and tables (if applicable)
- [x] All authors have ORCID identifiers
- [x] The paper is written in clear, correct English

## Checklist for Authors

Before submitting to JOSS, please ensure:

- [x] You've read the [JOSS submission requirements](https://joss.readthedocs.io/en/latest/submitting.html)
- [x] You've installed the [JOSS paper preview tool](https://github.com/openjournals/whedon) locally
- [x] The paper.md and paper.bib files are in the root directory of your repository
- [x] The paper.md file includes all required YAML header fields
- [x] All figures are correctly referenced in the paper text
- [x] The paper.bib file contains all necessary references in BibTeX format
- [x] You've created a release of your software with a DOI (e.g., via Zenodo)
- [x] The codemeta.json file is present and correctly filled out
