# HetServe-LLMs
This is a repository for organizing papers, codes and other resources related to the topic of distributed serving LLMs.

## LLM serving survey
- [Full Stack Optimization of Transformer Inference: a Survey](https://arxiv.org/abs/2302.14017) | UC Berkeley
- [Towards Efficient Generative Large Language Model Serving: A Survey from Algorithms to Systems](https://arxiv.org/pdf/2312.15234) | Carnegie Mellon University

## LLM serving

### Serving Metrics

### Latency-oriented
- [NeurIPS 2022] [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://proceedings.neurips.cc/paper_files/paper/2022/file/67d57c32e20fd0a7a302cb81d36e40d5-Paper-Conference.pdf) | Stanford University
- [ICML 2022] [DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale](https://arxiv.org/abs/2201.05596) | Microsoft
- [arXiv 2023.11] [FlashDecoding++: Faster Large Language Model Inference on GPUs](https://arxiv.org/abs/2311.01282) | Tsinghua University & Infinigence-AI
- [EMNLP 2023] [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/pdf/2305.13245) | Google Research
- [MLSys 2024] [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://hanlab.mit.edu/projects/awq) | MIT
- [ICLR 2024] [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://openreview.net/forum?id=mZn2Xyh9Ec) | Princeton University

### Throughput-oriented
- [OSDI 2022] [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu) | Seoul National University
- [ICML 2023] [FlexGen: high-throughput generative inference of large language models with a single GPU](https://dl.acm.org/doi/10.5555/3618408.3619696) | Stanford Univeristy
- [SOSP 2023] [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://dl.acm.org/doi/abs/10.1145/3600006.3613165) | UC Berkeley
- [ICLR 2024] [Efficient Streaming Language Models with Attention Sinks](https://hanlab.mit.edu/projects/streamingllm) | MIT
- [arXiv 2024.01] [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving](https://arxiv.org/pdf/2401.09670) | Peking University

### Distributed serving
- [Fast Distributed Inference Serving for Large Language Models](https://arxiv.org/abs/2305.05920) | Peking University
- [Infinite-LLM: Efficient LLM Service for Long Context with DistAttention and Distributed KVCache](https://arxiv.org/abs/2401.02669) | Alibaba Group
- [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu) | Seoul National University

### Open-source LLM serving systems
> tensor parallelism (TP), pipeline parallelism (PP), CPU-GPU offloading (offload)

|Serving system|Optimization target|Optimization|Parallel computation|Heterogeneous|
|:---|:---|:---|:---|:---|
|[FlexGen](https://github.com/FMInference/FlexGen)![Github stars](https://img.shields.io/github/stars/FMInference/FlexGen.svg) ![Github forks](https://img.shields.io/github/forks/FMInference/FlexGen.svg)|xxx|xxx|xxx|xxx|
|[Accelerate](https://github.com/huggingface/accelerate)![Github stars](https://img.shields.io/github/stars/huggingface/accelerate.svg) ![Github forks](https://img.shields.io/github/forks/huggingface/accelerate.svg)|xxx|xxx|xxx|xxx|
|[vLLM](https://github.com/vllm-project/vllm) ![Github stars](https://img.shields.io/github/stars/vllm-project/vllm.svg) ![Github forks](https://img.shields.io/github/forks/vllm-project/vllm.svg)|xxx|xxx|xxx|xxx|
|[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) ![Github stars](https://img.shields.io/github/stars/NVIDIA/TensorRT-LLM.svg) ![Github forks](https://img.shields.io/github/forks/NVIDIA/TensorRT-LLM.svg)|xxx|xxx|xxx|xxx|
|[LightLLM](https://github.com/ModelTC/lightllm) ![Github stars](https://img.shields.io/github/stars/ModelTC/lightllm.svg) ![Github forks](https://img.shields.io/github/forks/ModelTC/lightllm.svg)|xxx|xxx|xxx|xxx|
|[MLC-LLM](https://github.com/mlc-ai/mlc-llm) ![Github stars](https://img.shields.io/github/stars/mlc-ai/mlc-llm.svg) ![Github forks](https://img.shields.io/github/forks/mlc-ai/mlc-llm.svg)|xxx|xxx|xxx|xxx|
|[DeepSpeed](https://github.com/microsoft/DeepSpeed) ![Github stars](https://img.shields.io/github/stars/microsoft/DeepSpeed.svg) ![Github forks](https://img.shields.io/github/forks/microsoft/DeepSpeed.svg)|xxx|xxx|xxx|xxx|
|[llama.cpp](https://github.com/ggerganov/llama.cpp) ![Github stars](https://img.shields.io/github/stars/ggerganov/llama.cpp.svg) ![Github forks](https://img.shields.io/github/forks/ggerganov/llama.cpp.svg)|xxx|xxx|xxx|xxx|

## Serving on heterogeneous devices
- [PipeEdge: Pipeline Parallelism for Large-Scale Model Inference on Heterogeneous Edge Devices](https://github.com/usc-isi/PipeEdge) | Purdue University
- [LLM in a flash: Efficient Large Language Model Inference with Limited Memory](https://arxiv.org/abs/2312.11514) | Apple

## Other list
- [Awesome AI System](https://github.com/lambda7xx/awesome-AI-system)
- [Awesome LLM Systems Papers](https://github.com/AmberLJC/LLMSys-PaperList)
- [Awesome-LLM-System-Papers](https://github.com/AmadeusChan/Awesome-LLM-System-Papers?tab=readme-ov-file)

## Userful external links
- [Efficient Large Language Model (LLM) Inferencing on GPUs](https://github.com/yinuotxie/Efficient-LLM-Inferencing-on-GPUs?tab=readme-ov-file) 
