# HetServe-LLMs
This is a repository for organizing papers, codes and other resources related to the topic of distributed serving LLMs.

## LLM Serving Survey
- [Full Stack Optimization of Transformer Inference: a Survey](https://arxiv.org/abs/2302.14017) | UC Berkeley
- [Towards Efficient Generative Large Language Model Serving: A Survey from Algorithms to Systems](https://arxiv.org/pdf/2312.15234) | Carnegie Mellon University

## LLM Serving Systems

- Parallel compuation: tensor parallelism (TP), pipeline parallelism (PP), and CPU-GPU offloading (Offload).
- Heterogeneous: running LLMs on multiple heterogeneous devices. 

|Serving System|Prioritized Optimization Target|Serving Optimization|Main Features|Parallel Computation|Heterogeneous|
|:---:|:---:|:---:|:---:|:---:|
|[Accelerate](https://github.com/huggingface/accelerate)|xxx|xxx|xxx|✔️TP<br>✔️PP<br>✔️Offload||
|[vLLM](https://github.com/vllm-project/vllm)|Throughput|xxx|xxx|✔️TP<br>✔️PP|✔️|
|[llama.cpp](https://github.com/ggerganov/llama.cpp)|Latency|xxx|xxx|✔️TP<br>✔️PP|✔️|
|[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)|Latency|xxx|xxx|||
|[DeepSpeedInference](https://github.com/microsoft/DeepSpeed)|Throughput|xxx|xxx|||
|[LightLLM](https://github.com/ModelTC/lightllm)|xxx|xxx|xxx|||
|[MLC-LLM](https://github.com/mlc-ai/mlc-llm)|Latency|xxx|xxx|||

## Distributed Inference
- [PipeEdge: Pipeline Parallelism for Large-Scale Model Inference on Heterogeneous Edge Devices](https://github.com/usc-isi/PipeEdge) | Purdue
- [Fast Distributed Inference Serving for Large Language Models](https://arxiv.org/abs/2305.05920) | Peking University
- [Infinite-LLM: Efficient LLM Service for Long Context with DistAttention and Distributed KVCache](https://arxiv.org/abs/2401.02669) | Alibaba Group
