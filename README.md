# HetServe-LLMs
This is a repository for organizing papers, codes and other resources related to the topic of efficiently serving LLMs over heterogeneous devices.

## LLM serving survey
- [arXiv 2023.02] [Full Stack Optimization of Transformer Inference: a Survey](https://arxiv.org/abs/2302.14017) | UC Berkeley
- [arXiv 2023.12] [Towards Efficient Generative Large Language Model Serving: A Survey from Algorithms to Systems](https://arxiv.org/pdf/2312.15234) | Carnegie Mellon University

## LLM serving

![](https://github.com/yinuotxie/Efficient-LLM-Inferencing-on-GPUs/raw/main/media/llm_inferece_dataflow.png)
*Image source: [Efficient Large Language Model (LLM) Inferencing on GPUs](https://github.com/yinuotxie/Efficient-LLM-Inferencing-on-GPUs?tab=readme-ov-file)*

![](https://mmbiz.qpic.cn/mmbiz_jpg/AAQtmjCc74AzI4S2pibIpCrWfDWiaBeFZOibkVAQgVwI2m6IicEdEjHgPHTwR6aGclubTx9MRFVFoYSjObI9QgvQzA/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)
*Image source: [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://dl.acm.org/doi/abs/10.1145/3600006.3613165)*

### Background
- [NeurIPS 2017] [Attention is all you need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) | Google Brain
- [NeurIPS 2020] [Language Models are Few-Shot Learners](https://papers.nips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf) | OpenAI
- [arXiv 2020.01] [Scaling Laws for Neural Language Models](https://arxiv.org/pdf/2001.08361) | Johns Hopkins University and OpenAI
- [arXiv 2022.01] [Scaling Language Models: Methods, Analysis & Insights from Training Gopher](https://arxiv.org/pdf/2112.11446) | DeepMind

![](https://mmbiz.qpic.cn/mmbiz_png/AAQtmjCc74AzI4S2pibIpCrWfDWiaBeFZO6xKDZNW06WkyAfGByibk2iacC5bJkKbrzskicG84C1dM7d74iaMoRJXoQA/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)
*Image source: [Large Language Models (in 2023)](https://www.youtube.com/watch?v=dbo3kNKPaUA)*

### Serving Metrics
- [Tech Blog] [LLM Inference Performance Engineering: Best Practices](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices) | Mosaic AI Research
- [arXiv 2024.04] [Andes: Defining and Enhancing Quality-of-Experience in LLM-Based Text Streaming Services](https://arxiv.org/pdf/2404.16283) | University of Michigan

### Latency-oriented

#### Efficient models
- [ICML 2022] [DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale](https://arxiv.org/abs/2201.05596) | Microsoft
- [EMNLP 2023] [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/pdf/2305.13245) | Google Research
- [ICML 2023] [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) | Google Research
- [ACL 2023] [Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes](https://arxiv.org/abs/2305.02301) | University of Washington

#### Efficient operators/kernels
- [NeurIPS 2022] [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://proceedings.neurips.cc/paper_files/paper/2022/file/67d57c32e20fd0a7a302cb81d36e40d5-Paper-Conference.pdf) | Stanford University
- [ICLR 2024] [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://openreview.net/forum?id=mZn2Xyh9Ec) | Princeton University
- [arXiv 2023.11] [FlashDecoding++: Faster Large Language Model Inference on GPUs](https://arxiv.org/abs/2311.01282) | Tsinghua University & Infinigence-AI

#### Quantization
- [ICML 2023] [moothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://github.com/mit-han-lab/smoothquant) | MIT
- [MLSys 2024] [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://hanlab.mit.edu/projects/awq) | MIT

### Throughput-oriented

#### Resource management
- [OSDI 2022] [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu) | Seoul National University
- [ICML 2023] [FlexGen: high-throughput generative inference of large language models with a single GPU](https://dl.acm.org/doi/10.5555/3618408.3619696) | Stanford Univeristy
- [SOSP 2023] [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://dl.acm.org/doi/abs/10.1145/3600006.3613165) | UC Berkeley
- [ICLR 2024] [Efficient Streaming Language Models with Attention Sinks](https://hanlab.mit.edu/projects/streamingllm) | MIT
- [arXiv 2024.01] [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving](https://arxiv.org/pdf/2401.09670) | Peking University
- [arXiv 2024.02] [FlexLLM: A System for Co-Serving Large Language Model Inference and Parameter-Efficient Finetuning](https://arxiv.org/pdf/2402.18789) | Carnegie Mellon University
- [arXiv 2024.03] [AttentionStore: Cost-effective Attention Reuse across Multi-turn Conversations in Large Language Model Serving](https://arxiv.org/pdf/2403.19708) | National University of Singapore
- [arXiv 2024.04] [MuxServe: Flexible Multiplexing for Efficient Multiple LLM Serving](https://arxiv.org/abs/2404.02015) | The Chinese University of Hong Kong
- [arXiv 2024.04] [LoongServe: Efficiently Serving Long-context Large Language Models with Elastic Sequence Parallelism](https://arxiv.org/pdf/2404.09526) | Peking University
- [arXiv 2024.04] [BlockLLM: Multi-tenant Finer-grained Serving for Large Language Models](https://arxiv.org/pdf/2404.18322) | City University of Hong Kong
- [arXiv 2024.05] [vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention](https://arxiv.org/pdf/2405.04437) | Microsoft Research India

#### Parallelism
- [NeurIPS 2019] [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965) | Google
- [SC 2021] [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473) | NVIDIA
- [OSDI 2022] [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu) | Seoul National University
- [arXiv 2023.05] [Fast Distributed Inference Serving for Large Language Models](https://arxiv.org/abs/2305.05920) | Peking University
- [arXiv 2024.01] [Infinite-LLM: Efficient LLM Service for Long Context with DistAttention and Distributed KVCache](https://arxiv.org/abs/2401.02669) | Alibaba Group

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
- [FMEC 2023] [PipeEdge: Pipeline Parallelism for Large-Scale Model Inference on Heterogeneous Edge Devices](https://github.com/usc-isi/PipeEdge) | Purdue University
- [ASPLOS 2023] [STI: Turbocharge NLP Inference at the Edge via Elastic Pipelining](https://arxiv.org/abs/2207.05022) | University of Virginia
- [arXiv 2023.12] [LLM in a flash: Efficient Large Language Model Inference with Limited Memory](https://arxiv.org/abs/2312.11514) | Apple
- [arXiv 2023.12] [SuperServe: Fine-Grained Inference Serving for Unpredictable Workloads](https://arxiv.org/pdf/2312.16733) | Georgia Tech
- [arXiv 2024.02] [APISERVE: Efficient API Support for Large-Language Model Inferencing](https://arxiv.org/pdf/2402.01869) | University of California, San Diego

## Other list
- [Awesome AI System](https://github.com/lambda7xx/awesome-AI-system)
- [Awesome LLM Systems Papers](https://github.com/AmberLJC/LLMSys-PaperList)
- [Awesome-LLM-System-Papers](https://github.com/AmadeusChan/Awesome-LLM-System-Papers?tab=readme-ov-file)
- [Efficient Large Language Model (LLM) Inferencing on GPUs](https://github.com/yinuotxie/Efficient-LLM-Inferencing-on-GPUs?tab=readme-ov-file)
- [Numbers every LLM Developer should know](https://github.com/ray-project/llm-numbers)
- [Tensor Parallelism and Sequence Parallelism: Detailed Analysis](https://insujang.github.io/2024-01-11/tensor-parallelism-and-sequence-parallelism-detailed-analysis/)
