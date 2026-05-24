# Cloud Training Options for Unity ML-Agents

## Summary

No pure "upload build → click train" managed service exists except Unity's own ML-Agents Cloud (new, barely documented). All other options require some infrastructure setup.

## Options

### Unity ML-Agents Cloud (Official)
- **What**: Unity's managed cloud training — submit training sessions, auto-scales across machines, no Python/config needed
- **Cost**: Uses Unity plan credits; usage billing expected 2026
- **Maturity**: Newly launched, very limited public documentation
- **URL**: create.unity3d.com/ml-agents-cloud-training
- **Verdict**: Most promising long-term but not ready for serious use yet

### Google Colab (Free)
- **What**: Jupyter notebook with free GPU, clone ML-Agents repo and train
- **Cost**: Free
- **Maturity**: Well-documented community resources
- **Limitations**: Session time limits, can't run background jobs reliably
- **Best for**: Quick prototyping, learning, small runs
- **Community repo**: github.com/dhyeythumar/ML-Agents-with-Google-Colab

### AWS SageMaker RL
- **What**: Wrap Unity environment with UnityToGymWrapper, train with Ray RLlib on EC2 GPU instances
- **Cost**: ~$0.90+/hour (p2.xlarge)
- **Maturity**: High — officially documented by both AWS and Unity
- **Best for**: Production training runs
- **Docs**: github.com/Unity-Technologies/ml-agents/blob/main/docs/Training-on-Amazon-Web-Service.md

### Ray RLlib + Anyscale
- **What**: Distributed RL framework, wrap Unity env with Gymnasium adapter, scale across workers
- **Cost**: Free (self-hosted Ray) or usage-based (Anyscale managed)
- **Maturity**: High — actively maintained
- **Best for**: Distributed/parallel training, hyperparameter search (Ray Tune)

### Azure VMs
- **What**: N-Series GPU VMs with manual setup
- **Cost**: Variable
- **Maturity**: Works but documentation partially deprecated
- **Docs**: unity-technologies.github.io/ml-agents/Training-on-Microsoft-Azure/

### Hugging Face Hub
- **What**: Model sharing and distribution — NOT training
- **Cost**: Free for public models
- **CLI**: `mlagents-push-to-hf` and `mlagents-load-from-hf` are built into ML-Agents 4.x
- **Use**: Share trained `.onnx` models, download community pretrained models

## Recommended Path

| Goal | Solution |
|---|---|
| Free experimentation | Google Colab |
| Serious training runs | AWS SageMaker RL |
| Distributed / hyperparameter search | Ray RLlib |
| Share/distribute trained models | Hugging Face Hub |
| Watch for future | Unity ML-Agents Cloud |

## GPU vs CPU VMs

**Use CPU-only VMs.** For ML-Agents locomotion training, GPU instances are unnecessary and wasteful.

The neural network is a tiny 2-layer MLP (256 hidden units). Training update time is negligible. The bottleneck is always PhysX CPU simulation — the same reason a local RTX 2070 Super sits at 100% GPU utilization *waiting for Unity*, not doing heavy compute.

More CPU cores → more parallel environments via `--num-envs` → faster training wall-clock time.

| Instance | vCPUs | GPU | Cost (AWS) | `--num-envs` |
| --- | --- | --- | --- | --- |
| `c5.9xlarge` | 36 | None | ~$1.53/hr | 36 |
| `c5.18xlarge` | 72 | None | ~$3.06/hr | 72 |
| `p3.2xlarge` | 8 | V100 | ~$3.06/hr | 8 |

The 36-vCPU CPU VM trains ~4.5× faster than the V100 instance at half the cost for this workload.

GPU instances only help if you switch to image-based observations (CNN) or transformer policies — not standard MLP locomotion agents.

## Notes

- Unity Gaming Services (multiplayer, analytics) does NOT support ML-Agents training
- Docker support is officially deprecated in recent ML-Agents versions
- All cloud solutions still require a headless Unity build — same `.exe` as local headless training
- The fundamental bottleneck (PhysX single-threaded) still applies on cloud VMs — more CPU cores help via `--num-envs`, not faster single-core speed
