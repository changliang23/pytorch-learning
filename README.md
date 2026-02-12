流程题
1. 构造语料
2. Tokenizer
3. 定义 Transformer
4. 预训练 (LM)
5. SFT (指令微调)
6. RLHF (奖励模型 + PPO)
7. 推理
8. 保存导出

目录结构
pytorch-learning/
 ├── data_pretrain.txt
 ├── data_sft.json
 ├── tokenizer.py
 ├── model.py
 ├── pretrain.py
 ├── sft.py
 ├── rlhf.py
 ├── infer.py
 ├── export.py
 └── main.py

DAY 1(26.2.12)
基础版，跑通全流程

DAY 2
Transformer 多头注意力版


DAY 3 
LoRA 微调版

DAY 4
加入 RLHF（PPO）版