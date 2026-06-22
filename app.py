import os
import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Convert the local directory into an absolute path to bypass URI path checks
model_path = os.path.abspath("savedbertmodel")

# Load your custom saved fine-tuned model and tokenizer using the absolute path
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(
    model_path, local_files_only=True
)

# Map the numeric labels back to the official AG News category strings
categories = ["World News", "Sports", "Business", "Sci/Tech"]


# Define the prediction function for the web interface
def classify_news(headline):
    # Convert text input into tokens
    inputs = tokenizer(
        headline, return_tensors="pt", padding=True, truncation=True
    )

    # Disable gradient tracking to make calculation faster
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract the probability scores
    scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
    scores = scores.numpy()[0]

    # Return a dictionary containing the category name and its score
    return {categories[i]: float(scores[i]) for i in range(4)}


# Create the Gradio interface layout
app = gr.Interface(
    fn=classify_news,
    inputs=gr.Textbox(lines=2, placeholder="Type a news headline here..."),
    outputs=gr.Label(num_top_classes=3),
    title="News Topic Classifier Using Fine-Tuned BERT",
    description="Type any headline to see how the model categorizes it.",
)

# Launch the local web server
if __name__ == "__main__":
    app.launch()