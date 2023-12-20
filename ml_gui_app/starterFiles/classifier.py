from taipy.gui import Gui
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the pre-trained model
model = load_model("baseline.keras")

classes_name = {
    0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
    5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"
}

def predict_image(model, path_to_image):
    try:
        img = Image.open(path_to_image)
        img = img.convert("RGB")
        img = img.resize((32, 32))
        data = np.asarray(img) / 255.0  # Normalize data
        probs = model.predict(np.array([data]))[0]
        top_prob = np.max(probs)
        top_pred = classes_name[np.argmax(probs)]
        return top_prob, top_pred
    except Exception as e:
        print(f"Error during prediction: {e}")
        return 0, "Error"

# Initialize variables for GUI content
content = ""
img_path = "placeholder_image.png"
prob = 0
pred = ""

index = """
<|text-center|
<|{"logo.png"}|image|width=25vw|>

<|{content}|file_selector|extensions=.png|>
Select an image from your file system

<|{pred}|>

<|{img_path}|image|>

<|{prob}|indicator|value={prob}|min=0|max=100|width=25vw|>
>
"""

def on_change(state, var_name, var_val):
    if var_name == "content":
        try:
            top_prob, top_pred = predict_image(model, var_val)
            state.prob = round(top_prob * 100)
            state.pred = 'This is a ' + top_pred
            state.img_path = var_val
        except Exception as e:
            print(f"Error: {e}")

# Initialize the GUI with default values
app = Gui(page=index)

# Run the app
if __name__ == "__main__":
    app.run(use_reloader=True, port=5050)

