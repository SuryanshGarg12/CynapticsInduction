# Fine-Tuning Phi-2 for Persona-Chat

This repository contains scripts for fine-tuning the [Phi-2 language model](https://huggingface.co/microsoft/phi-2) on the [Persona-Chat dataset](https://huggingface.co/datasets/Cynaptics/persona-chat). The goal is to make Phi-2 better at generating responses that align with specific personas in conversations.

## Overview

I fine-tuned Phi-2 using LoRA (Low-Rank Adaptation) and 4-bit quantization to make the training process efficient and lightweight while still achieving great performance. Here's the general approach:

1. **Dataset**: We used Persona-Chat, which provides persona-based dialogues.
2. **Fine-Tuning**: Applied LoRA with 4-bit quantization to save memory and speed up training.
3. **Model**: The fine-tuned Phi-2 model is now more aligned with persona-based conversations and can generate personalized responses.

---

## Steps

1. **Install Required Libraries**  
   We used popular libraries like `transformers`, `datasets`, and `peft` for efficient fine-tuning. Install them using the commands in the script.

2. **Preprocessing**  
   The Persona-Chat dataset was reformatted to create prompts like: Instruct: Given this persona: <persona> and this dialogue: <dialogue> generate a response. Output: <reference response>


3. **Quantization**  
Used 4-bit quantization to reduce memory usage while maintaining accuracy.

4. **LoRA Fine-Tuning**  
Fine-tuned specific layers (`q_proj`, `k_proj`, etc.) using LoRA, a lightweight method for adapting large language models.

5. **Training**  
Trained the model for 1 epoch (500 steps) using a small batch size and cosine learning rate scheduler.

6. **Evaluation**  
Generated responses to test prompts to validate the model's ability to respond in a persona-aligned way.

## Why These Choices?

- **Phi-2**: Chosen for its lightweight design and strong performance on conversational tasks.
- **LoRA**: Efficient and resource-friendly for fine-tuning large language models.
- **4-bit Quantization**: Drastically reduced memory requirements without compromising output quality.



