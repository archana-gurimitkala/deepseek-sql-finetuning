"""
DeepSeek SQL - Text-to-SQL with fine-tuned DeepSeek Coder 1.3B (LoRA)
"""

import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "deepseek-ai/deepseek-coder-1.3b-instruct"
LORA_MODEL = "Archanacreates/deepseek-sql-lora"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
    trust_remote_code=True,
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, LORA_MODEL)
model.eval()
print("Model ready!")


def generate_sql(question, context=""):
    if not question.strip():
        return ""

    prompt = f"""### Instruction:
Convert this question to SQL.

### Input:
{question}
{context}

### Response:
"""

    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    new_tokens = outputs[0][input_len:]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
    decoded = decoded.replace("Ġ", " ").replace("Ċ", "\n").strip()

    # Stop at semicolon if present
    if ";" in decoded:
        return decoded.split(";")[0].strip() + ";"

    # Stop at any new SQL statement keyword
    import re
    match = re.search(r'\b(CREATE|INSERT|DROP|ALTER|USE|END|##)\b', decoded, re.IGNORECASE)
    if match:
        sql = decoded[:match.start()].strip()
    else:
        sql = decoded.split("\n")[0].strip()

    return sql if sql else decoded.strip()


with gr.Blocks(title="DeepSeek SQL") as demo:
    gr.Markdown("""
# DeepSeek SQL — Text to SQL
Fine-tuned **DeepSeek Coder 1.3B** with LoRA on 5,000 text-to-SQL pairs.
Converts natural language questions into SQL queries — no proprietary API needed.
""")

    with gr.Row():
        with gr.Column():
            question = gr.Textbox(
                label="Your Question",
                placeholder="Show all customers from New York",
                lines=2,
            )
            context = gr.Textbox(
                label="Database Schema (optional)",
                placeholder="CREATE TABLE customers (id INT, name VARCHAR, city VARCHAR)",
                lines=4,
            )
            with gr.Row():
                clear_btn = gr.Button("Clear", variant="secondary")
                submit_btn = gr.Button("Generate SQL", variant="primary")

        with gr.Column():
            sql_output = gr.Textbox(label="Generated SQL", lines=5, interactive=False)

    gr.Examples(
        examples=[
            ["Show all customers", "CREATE TABLE customers (id INT, name VARCHAR, city VARCHAR)"],
            ["Find products cheaper than $50", "CREATE TABLE products (id INT, name VARCHAR, price DECIMAL)"],
            ["Count orders by status", "CREATE TABLE orders (id INT, customer_id INT, status VARCHAR)"],
            ["Show top 5 most expensive products", "CREATE TABLE products (id INT, name VARCHAR, price DECIMAL)"],
        ],
        inputs=[question, context],
    )

    gr.Markdown("""
---
**Model:** DeepSeek Coder 1.3B + LoRA fine-tuning | **Accuracy:** 10% exact match, 42% token accuracy
**Training:** 5,000 examples, 5 epochs on Kaggle T4 GPU
""")

    submit_btn.click(fn=generate_sql, inputs=[question, context], outputs=sql_output)
    clear_btn.click(fn=lambda: ("", "", ""), outputs=[question, context, sql_output])

if __name__ == "__main__":
    demo.launch()
