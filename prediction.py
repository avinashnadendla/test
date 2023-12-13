import joblib
# Loading the model from the file
loaded_model = joblib.load("text_classification_model.joblib")


def predict_category(item, model):
    return model.predict([item])[0]

# Use the loaded model for prediction
example_item = "ENSURE REG VAN 6PK 8Z"
predicted_category = predict_category(example_item, loaded_model)
print(f"The give item {example_item} is of type {predicted_category}")
