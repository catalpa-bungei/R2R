---
library_name: r2r
pipeline_tag: text-classification
tags:
- router
- efficiency
- language-model
---

# R2R Router Models

This repository provides a collection of **R2R** routers (Mixture of Small and Large Language Models) and its training config built for different model pairs.

## Model Description

R2R routers are lightweight classifiers that decide, at the token level, whether to generate with a small language model (SLM) or delegate to a large language model (LLM). The goal is to retain LLM-level quality while improving end-to-end efficiency.

We currently support routers for the **Qwen3** series and the **DeepSeek-R1-Qwen** series under *deterministic (non-sampling)* decoding. In addition, we provide a router tailored for routing between **DeepSeek-R1-Qwen-1.5B** and **DeepSeek-R1-Qwen-32B** under DeepSeek’s *default sampling* settings(temperature=0.6, top_p=0.95).

## Usage

For setup instructions, checkpoints, and examples, please visit our GitHub repository:

- GitHub: [https://github.com/thu-nics/R2R](https://github.com/thu-nics/R2R)  
- Project page: [https://fuvty.github.io/R2R_Project_Page/](https://fuvty.github.io/R2R_Project_Page/)
