
from django.urls import path, include
from  . import views
from django.contrib import admin
from django.conf import settings
from django.conf.urls.static import static
urlpatterns = [
    path('',views.home ,name='home'),
    path('home',views.home ,name='home'),
    path('about', views.About, name='about'),
    path('form', views.Product_form, name='form'),
    path('by_image', views.by_image, name='by_image'),
]
