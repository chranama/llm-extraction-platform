# AWS EKS vLLM Backend Overlay

This overlay is the promoted live-model AWS workflow for the backend. It keeps
the LLMEP API and worker on CPU nodes and adds a separate vLLM model-runtime
Deployment that the backend reaches through the OpenAI-compatible HTTP API.

The deterministic fake workflow remains in `deploy/k8s/overlays/aws-eks/`.

## Runtime Contract

- `api` and `extract-worker` use the slim `llm-server` image.
- `vllm` uses the public `vllm/vllm-openai` image and runs on a GPU node.
- `models.aws-vllm.yaml` selects `backend: remote` with `provider: vllm`,
  which reaches the external OpenAI-compatible runtime over HTTP.
- The vLLM pod requires a node labeled `workload=model-runtime` and
  `accelerator=nvidia`, plus a matching `workload=model-runtime:NoSchedule`
  toleration.

Before applying this overlay, the AWS substrate must include a GPU node group
and an NVIDIA device-plugin path appropriate for the EKS cluster.
