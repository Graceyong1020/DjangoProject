from django.shortcuts import render
from imagePro.imageTest import data_process

# Create your views here.
def imageProcess(request):
    path = request.GET.get('path', '')  # Get the 'path' parameter from the request
    predict=data_process(path)
    return render(request, 'imagePro/imageTest.html', {'predict':predict})

