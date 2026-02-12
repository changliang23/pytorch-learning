from pretrain import run_pretrain
from sft import run_sft
from infer import run_infer

def main():
    print("🚀 Step1: Pretraining...")
    tok1 = run_pretrain()
    print("✅ Pretrain finished")

    print("🎯 Step2: SFT...")
    tok2 = run_sft()
    print("✅ SFT finished")

    print("🤖 Step3: Inference...")
    result = run_infer("你好")
    print("🧠 模型输出:", result)

if __name__ == "__main__":
    main()
