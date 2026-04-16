
## 💻 How to Install
You can also refer to [llm2vec](https://github.com/McGill-NLP/llm2vec)
```
conda create -n llm2vec  python=3.10 -y
conda activate llm2vec
pip install llm2vec
pip install flash-attn --no-build-isolation
pip install deepspeed
pip install accelerate==0.34.2 # the scripts provided by llm2vec can't be directly runned in the newest accelerate
```
### 🔥 Training
We train all the models in 8*80g h100.  
For mntp with cc3m
1. Prepare cc3m with key short_caption and long_caption in csv format
2. cc3m.csv path in MetaLlama3_cc3m.json

And you can train whth the following scripts
```cd llm2vec
HF_TOKEN=xxxx accelerate launch --config_file ./ac_zero2.yaml run_mntp.py train_configs/mntp/MetaLlama3_cc3m.json  
```
For supervised with cc3m
1. First prepare e5 data used in [llm2vec](https://github.com/McGill-NLP/llm2vec) 
,also prepare the same cc3m csv
2. Add the lora weights pretrained in mntp in train_configs/supervised/MetaLlama3_cc3m.json

And you can train with the following scripts
```
HF_TOKEN=xxxx --config_file ./ac_zero2.yaml run_supervised.py  train_configs/supervised/MetaLlama3_cc3m.json
```

## ❤️ Acknowlegement

This code is built on top of [llm2vec](https://github.com/McGill-NLP/llm2vec). Thanks for their nice work!