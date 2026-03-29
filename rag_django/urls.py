from django.contrib import admin
from django.urls import path, include

urlpatterns = [ 

    path('admin/', admin.site.urls),
    path('', include('rag_app.urls')),  #   Mọi URL từ trang chủ, sẽ chuyển sang rag_app.urls xử lý

]