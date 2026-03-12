Project Description
Workflow
1. Prepare data  
2. Tokenizer  
3. Transformer Definition  
4. Pre-training (LM)  
5. SFT (Instruction Fine-tuning)  
6. RLHF (Reward Model + PPO)  
7. Inference  
8. Model Export  

Directory Structure  
pytorch-learning/
 ├── data_pretrain.txt    # Pre-training data  
 ├── data_sft.json        # Instruction fine-tuning data  
 ├── tokenizer.py         # Tokenizer  
 ├── model.py             # Mini GPT model definition  
 ├── pretrain.py          # Pre-training script  
 ├── sft.py               # Instruction fine-tuning script  
 ├── rlhf.py              # RLHF training script  
 ├── infer.py             # Inference script  
 ├── export.py            # Model export script  
 └── main.py              # Main entry point  

Development Plan  
DAY 1 (Feb 12, 2026)  
Basic Version - Complete the full pipeline

DAY 2  
Multi-head Attention Transformer - Implement complete multi-head attention mechanism

DAY 3  
LoRA Fine-tuning Version - Add Low-Rank Adaptation fine-tuning

DAY 4  
RLHF Version - Add Reinforcement Learning from Human Feedback based on PPO

