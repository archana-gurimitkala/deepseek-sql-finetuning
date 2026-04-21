---
title: DeepSeek SQL Fine-tuning
emoji: 🗃️
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "5.25.0"
app_file: app.py
pinned: false
---

# DeepSeek SQL — Text to SQL with Open-Source Fine-tuning

Fine-tuned **DeepSeek Coder 1.3B** with LoRA to convert natural language into SQL — no proprietary API needed.

## How It Works

1. Base model: `deepseek-ai/deepseek-coder-1.3b-instruct`
2. Fine-tuned with LoRA on 5,000 text-to-SQL pairs
3. Trained on Kaggle T4 GPU for ~8 hours

## Results

| Metric | Score |
|--------|-------|
| Exact Match Accuracy | 10% |
| Token Accuracy | 42.38% |
| Training Loss | 5.74 → 0.21 |
