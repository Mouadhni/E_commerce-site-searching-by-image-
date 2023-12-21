import os

from django.shortcuts import render, redirect

# Create your views here.
from  django.http import HttpResponse

from django_Project import settings
from myapp.forms import Productform
from myapp.models import Product,Category
from django.shortcuts import render
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import GlobalMaxPooling2D
import cv2
import numpy as np
from numpy.linalg import norm
import pickle
from sklearn.neighbors import NearestNeighbors
import tensorflow
# Load the pre-existing feature list and filenames
feature_vector_path = os.path.join(settings.STATICFILES_DIRS[0], 'featurevector.pkl')
filename_path = os.path.join(settings.STATICFILES_DIRS[0], 'filename.pkl')

# Load the files
feature_list = np.array(pickle.load(open(feature_vector_path, "rb")))
filename = pickle.load(open(filename_path, "rb"))

# Create ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(244, 244, 3))
model.trainable = False
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])
model.summary()

def by_image(request):
    if request.method == 'POST' and request.FILES['photo']:
        photo = request.FILES['photo']
        img = cv2.imdecode(np.fromstring(photo.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (244, 244))
        img = np.array(img)
        expanf_img = np.expand_dims(img, axis=0)
        pre_img = preprocess_input(expanf_img)
        result = model.predict(pre_img).flatten()
        normalized = result / norm(result)

        neighbors = NearestNeighbors(n_neighbors=10, algorithm='brute', metric="euclidean")
        neighbors.fit(feature_list)
        distance, indices = neighbors.kneighbors([normalized])
        print(indices)
        similar_files = [filename[file] for file in indices[0][1:10]]
        return render(request, 'by_image.html', {'similar_files': similar_files})

    return render(request, 'by_image.html')


def home(request):
    products = Product.objects.all()
    categories = Category.objects.all()

    # Pass the products and categories to the template context
    context = {'products': products, 'categories': categories}
    return render(request, "home.html", context)

def About(request):
    return  render(request, "about.html")


def Product_form(request):
    if request.method == 'POST':
        form = Productform(request.POST)
        if form.is_valid():
            form.save()
            # Redirect to a success page or re-render the form without initial data
            return redirect('form')  # Replace 'success' with your success URL or view name
    else:
        form = Productform()

    context = {'form': form}
    return render(request, 'form.html', context)