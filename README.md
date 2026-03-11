Text-to-SQL Generation with Fine-tuned DeepSeek Coder
Project Overview

After fine-tuning GPT-4o-mini, I wanted to see if I could get similar results using a fully open-source model with no API costs. I used DeepSeek Coder 1.3B with LoRA and 4-bit quantization to make training fit on a Kaggle T4 GPU — and watched the training loss drop from 5.74 down to 0.21. The experience of tuning a small open model from scratch was worth every bit of it.

  ---

Instead of relying on proprietary models like GPT-4 or Claude, I explored the open-source ecosystem to build a cost-effective and privacy-friendly solution that can run locally.

Problem Statement

Writing SQL queries requires technical knowledge. This project helps users ask questions in plain English and automatically generate SQL queries.

Input:
"Who is the opponent when the record was 0-1?"

Output:

SELECT "Opponent" FROM table WHERE "Record" = '0-1'

Technical Approach
Component	Choice	Why
Base Model	DeepSeek Coder 1.3B	Open-source, optimized for code, lightweight
Fine-tuning	LoRA (Low-Rank Adaptation)	Memory efficient and faster training
Quantization	4-bit (BitsAndBytes)	Allows training on limited GPU memory
Dataset	5,000 text-to-SQL pairs	Good balance for learning SQL patterns
Hardware	Kaggle T4 GPU	Free and accessible
Process

Model Selection — Chose an open-source model instead of proprietary APIs

Data Preparation — Formatted question, schema, and SQL query pairs

Memory Optimization — Used 4-bit quantization to fit GPU limits

Fine-tuning — Trained the model using LoRA adapters for 5 epochs

Evaluation — Tested the model on unseen examples

Results
Metric	Score
Exact Match Accuracy	10.00%
Token-level Accuracy	42.38%
Training Time	~8 hours
Training Loss	5.74 → 0.21

The large drop in training loss shows the model successfully learned SQL patterns from the dataset.

Key Learnings

Open-source models are powerful and highly customizable

LoRA and quantization make fine-tuning possible even with limited hardware

Proper evaluation metrics help measure progress and identify improvements

Try the Model

🤗 Hugging Face:
https://huggingface.co/Archanacreates/deepseek-sql-lora

Skills Demonstrated

LLM Fine-tuning • LoRA • Quantization • Hugging Face Transformers • PEFT • Model Evaluation • Python

Future Improvements

Increase dataset size to improve generalization

Experiment with larger base models

Build an interactive demo interface

This project marks my transition from using AI tools to building AI systems.                                                                                                                                                              
  This project marks my transition from using AI to building AI.                                                                                                                                                                                 
  ```                                                                              
