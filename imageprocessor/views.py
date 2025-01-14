from django.shortcuts import render
from .forms import ImageUploadForm
from .models import ImagePrediction
import json
import numpy as np
from PIL import Image

import torch.nn.functional as F
import pickle
import torch

from torchvision import transforms


# Классы для предсказаний
classes = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum',
           'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber',
           'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi',
           'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear',
           'peas', 'pineapple', 'pomegranate', 'potato', 'raddish',
           'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato',
           'turnip', 'watermelon']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


from .models import ResNet50Custom

with open('ResNet50Custom.pkl', 'rb') as f:
    model2 = pickle.load(f)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model2 = model2.to(device)
model2.eval()
# ЗАГЛУШКА для предсказаний
def mock_predict_image(image):
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model2(img_tensor)
        probabilities = F.softmax(outputs, dim=1)
        _, predicted_class = torch.max(outputs, 1)

    return probabilities.squeeze().cpu().numpy(), classes[
        predicted_class.item()]

# Обработка загрузки изображения
def upload_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            img = Image.open(image)
            predictions, predicted_class = mock_predict_image(img)

            # Сохраняем результаты в базу данных
            predictions_json = json.dumps(predictions.tolist())

            # Выводим топ-5 предсказаний
            top_predictions = np.argsort(predictions)[::-1][:5]
            top_classes = [classes[i] for i in top_predictions]
            top_probs = predictions[top_predictions]
            image_instance = ImagePrediction.objects.create(
                image=image,
                predictions=json.dumps(predictions.tolist())
            )

            # Получаем URL изображения из модели
            image_url = image_instance.image.url
            classes_and_probs = zip(top_classes, top_probs)

            return render(request, 'result.html',
                          {'classes_and_probs': classes_and_probs, 'image_url': image_url,})
    else:
        form = ImageUploadForm()


    return render(request, 'upload.html', {'form': form})

# Показ истории запросов
def history(request):
    history = ImagePrediction.objects.order_by('-uploaded_at')
    return render(request, 'history.html', {'history': history,"classes": classes})
