from django.urls import path
from . import views

urlpatterns = [
    path('', views.index), 
    path('upload/', views.upload_pdf),
    path('ask/', views.ask_question),
    path('history/', views.get_chat_history),
    path('delete-chat/', views.delete_chat),
    path('switch-document/', views.switch_document),
    path('create-chat/', views.upload_pdf),
    # user sẽ gửi request -> Django nhận request -> Tạo HttpRequest -> tìm hàm và nhét request mặc định vào
]