import os
from PIL import Image
import torch
import gradio as gr
from transformers import CLIPModel, CLIPProcessor

# Load OpenAI's lightweight CLIP model and its matching data processor
# This model maps text descriptions and image pixels into a shared vector space
model_name = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name)

print("CLIP Multimodal model loaded successfully!")


# Define the predictive zero-shot classification function
def classify_image(input_image, candidate_labels_text):
    # If the user leaves the labels blank, provide default choices
    if not candidate_labels_text.strip():
        labels = ["landscape", "animal", "food", "vehicle", "human"]
    else:
        # Split the comma-separated labels text into a clean list of words
        labels = [label.strip() for label in candidate_labels_text.split(",")]

    # Convert the raw image into RGB mode to ensure consistency
    if input_image.mode != "RGB":
        input_image = input_image.convert("RGB")

    # Extract features and preprocess both modalities simultaneously
    inputs = processor(
        text=labels, images=input_image, return_tensors="pt", padding=True
    )

    # Disable gradient computation to perform fast inferencing on CPU
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract the similarity logits between the image and the text array
    logits_per_image = outputs.logits_per_image

    # Apply a Softmax function to convert logits into clean percentages/probabilities
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

    # Map each text label to its calculated probability score
    return {labels[i]: float(probs[i]) for i in range(len(labels))}


# Build the clean Gradio browser layout
with gr.Blocks() as app:
    gr.Markdown("# Multimodal Image-Text Matching & Zero-Shot Classifier")
    gr.Markdown(
        "Upload any image and type custom categories separated by commas. "
        "The pre-trained CLIP model will automatically calculate the best match with no training required!"
    )

    with gr.Row():
        with gr.Column():
            # Input layout: Image upload box and custom label text field
            image_input = gr.Image(type="pil", label="Upload Image")
            labels_input = gr.Textbox(
                label="Candidate Categories (separated by commas)",
                placeholder="e.g., dog, cat, red sports car, plate of food",
                lines=2,
            )
            submit_btn = gr.Button("Analyze Image Match", variant="primary")

        with gr.Column():
            # Output layout: A clean label grid presenting the classification results
            output_labels = gr.Label(num_top_classes=5, label="Matching Scores")

    # Tie the submit button component to the processing function execution
    submit_btn.click(
        fn=classify_image,
        inputs=[image_input, labels_input],
        outputs=output_labels,
    )

# Launch the local environment web server module interface
if __name__ == "__main__":
    app.launch()