from django.shortcuts import render

# Create your views here.
from  django.http import HttpResponse, HttpResponseNotFound

def handler404(request, exception):
    return HttpResponse('404; page not found')
def Base(request):
    return render(request,'base.html')