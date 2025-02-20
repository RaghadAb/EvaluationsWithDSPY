# Comparative Analysis of Zero-Shot and X-Shot on Open-Source LLMs for DNS-Based Email Security with DSPy  
### Advancing DMARC Threat Assessment and Policy Optimization

## Masters Project

This repository accompanies the Master's project for the University of Leeds

## Objectives

- **Evaluate Prompting Strategies:** Compare zero-shot, few-shot (X-Shot), and Bootstrap Few-Shot (BFS) prompting strategies in the context of email security.
- **Assess LLM Performance:** Benchmark open-source LLMs on structured Q&A tasks related to DMARC, SPF, and DKIM.
- **Optimize Prompts Automatically:** Leverage DSPy to automatically refine and optimize prompts for improved reasoning and response quality.
- **Enhance Reproducibility:** Provide an open-source implementation to support the research paper and facilitate reproducibility of experiments.

## Repository Overview

This repository contains the code supporting the research paper. It is designed to:
- Enable reproducibility of experiments related to email security evaluations.
- Support running the experiments on Google Colab or locally.
- Allow users to integrate their own API keys (Langwatch and OpenAI) and download models via the Ollama framework.

**Note:** The dataset used in the paper is not included in this repository due to commercial restrictions. Users should substitute their own dataset.

## Getting Started

### Prerequisites

- **Python 3.7 or higher**
- Access to Langwatch and OpenAI API keys
- Ollama framework for downloading and managing models locally
