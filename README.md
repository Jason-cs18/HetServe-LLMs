# HetServe-LLMs
This is a repository for organizing papers, codes and other resources related to the topic of distributed serving LLMs.

## LLM serving survey
- [Full Stack Optimization of Transformer Inference: a Survey](https://arxiv.org/abs/2302.14017) | UC Berkeley
- [Towards Efficient Generative Large Language Model Serving: A Survey from Algorithms to Systems](https://arxiv.org/pdf/2312.15234) | Carnegie Mellon University

## LLM serving optimizations
### Deep compression
### KV Cache
### Decoding strategies
### Batch inference

## LLM serving systems
> tensor parallelism (TP), pipeline parallelism (PP), CPU-GPU offloading (offload)

|Serving system|Optimization target|Optimization|Main features|Parallel computation|Heterogeneous|
|:---|:---|:---|:---|:---|:---|
|[FlexGen](https://github.com/FMInference/FlexGen)![Github stars](https://img.shields.io/github/stars/FMInference/FlexGen.svg) ![Github forks](https://img.shields.io/github/forks/FMInference/FlexGen.svg)|xxx|xxx|xxx|xxx|xxx|
|[Accelerate](https://github.com/huggingface/accelerate)![Github stars](https://img.shields.io/github/stars/huggingface/accelerate.svg) ![Github forks](https://img.shields.io/github/forks/huggingface/accelerate.svg)|xxx|xxx|xxx|xxx|xxx|
|[vLLM](https://github.com/vllm-project/vllm) ![Github stars](https://img.shields.io/github/stars/vllm-project/vllm.svg) ![Github forks](https://img.shields.io/github/forks/vllm-project/vllm.svg)|xxx|xxx|xxx|xxx|xxx|
|[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) ![Github stars](https://img.shields.io/github/stars/NVIDIA/TensorRT-LLM.svg) ![Github forks](https://img.shields.io/github/forks/NVIDIA/TensorRT-LLM.svg)|xxx|xxx|xxx|xxx|xxx|
|[LightLLM](https://github.com/ModelTC/lightllm) ![Github stars](https://img.shields.io/github/stars/ModelTC/lightllm.svg) ![Github forks](https://img.shields.io/github/forks/ModelTC/lightllm.svg)|xxx|xxx|xxx|xxx|xxx|
|[MLC-LLM](https://github.com/mlc-ai/mlc-llm) ![Github stars](https://img.shields.io/github/stars/huggingface/accelerate.svg) ![Github forks](https://img.shields.io/github/forks/huggingface/accelerate.svg)|xxx|xxx|xxx|xxx|xxx|
|[DeepSpeed](https://github.com/microsoft/DeepSpeed) ![Github stars](https://img.shields.io/github/stars/microsoft/DeepSpeed.svg) ![Github forks](https://img.shields.io/github/forks/microsoft/DeepSpeed.svg)|xxx|xxx|xxx|xxx|xxx|

## Distributed inference
- [PipeEdge: Pipeline Parallelism for Large-Scale Model Inference on Heterogeneous Edge Devices](https://github.com/usc-isi/PipeEdge) | Purdue University
- [Fast Distributed Inference Serving for Large Language Models](https://arxiv.org/abs/2305.05920) | Peking University
- [Infinite-LLM: Efficient LLM Service for Long Context with DistAttention and Distributed KVCache](https://arxiv.org/abs/2401.02669) | Alibaba Group
