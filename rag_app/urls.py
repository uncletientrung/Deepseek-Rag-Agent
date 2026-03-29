from django.urls import path
from . import views

urlpatterns = [
    path('', views.index), 
    path('upload/', views.upload_pdf),
    path('ask/', views.ask_question),
    # user sẽ gửi request -> Django nhận request -> Tạo HttpRequest -> tìm hàm và nhét request mặc định vào
]