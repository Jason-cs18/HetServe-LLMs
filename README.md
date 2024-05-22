# HetServe-LLMs
This is a repository for organizing papers, codes and other resources related to the topic of distributed serving LLMs.

## LLM Serving Survey
- [Full Stack Optimization of Transformer Inference: a Survey](https://arxiv.org/abs/2302.14017) | UC Berkeley
- [Towards Efficient Generative Large Language Model Serving: A Survey from Algorithms to Systems](https://arxiv.org/pdf/2312.15234) | Carnegie Mellon University

## LLM Serving Systems
> tensor parallelism (TP), pipeline parallelism (PP), CPU-GPU offloading (offload)

- [Accelerate](https://github.com/huggingface/accelerate) | HuggingFace
  - Prioritized optimization target: xxx
  - Optimization: xxx
  - Main features: xxx
  - Parallel computation: TP, PP, offload
  - Heterogeneous: xxx 
- [vLLM](https://github.com/vllm-project/vllm) | UC Berkeley
  - Prioritized optimization target: xxx
  - Optimization: xxx
  - Main features: xxx
  - Parallel computation: xxx
  - Heterogeneous: xxx 
- [llama.cpp](https://github.com/ggerganov/llama.cpp) | Georgi Gerganov
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | NVIDIA
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) | Microsoft
- [LightLLM](https://github.com/ModelTC/lightllm) | ModelTC
- [MLC-LLM](https://github.com/mlc-ai/mlc-llm) | CMU

|Serving System|Prioritized Opt. Target|Optimization|Main Features|Parallel Computation|Heterogeneous|
|:---:|:---|:---|:---|:---|:---:|
|[Accelerate](https://github.com/huggingface/accelerate)|xxx|xxx|xxx|✔️TP<br>✔️PP<br>✔️Offload||
|[vLLM](https://github.com/vllm-project/vllm)|Throughput|xxx|xxx|✔️TP<br>✔️PP|✔️|
|[llama.cpp](https://github.com/ggerganov/llama.cpp)|Latency|xxx|xxx|✔️TP<br>✔️PP|✔️|
|[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)|Latency|xxx|xxx|||
|[DeepSpeedInference](https://github.com/microsoft/DeepSpeed)|Throughput|xxx|xxx|||
|[LightLLM](https://github.com/ModelTC/lightllm)|xxx|xxx|xxx|||
|[MLC-LLM](https://github.com/mlc-ai/mlc-llm)|Latency|xxx|xxx|||

## Distributed Inference
- [PipeEdge: Pipeline Parallelism for Large-Scale Model Inference on Heterogeneous Edge Devices](https://github.com/usc-isi/PipeEdge) | Purdue University
- [Fast Distributed Inference Serving for Large Language Models](https://arxiv.org/abs/2305.05920) | Peking University
- [Infinite-LLM: Efficient LLM Service for Long Context with DistAttention and Distributed KVCache](https://arxiv.org/abs/2401.02669) | Alibaba Group
