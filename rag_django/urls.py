from django.contrib import admin
from django.urls import path, include
from django.views.static import serve
from django.conf import settings
import os

urlpatterns = [ 

    path('admin/', admin.site.urls),
    path('', include('rag_app.urls')),  #   Mọi URL từ trang chủ, sẽ chuyển sang rag_app.urls xử lý
    # Serve file PDF/DOCX từ thư mục data/ để pdf.js có thể đọc được
    path('data/<path:path>', serve, {
        'document_root': os.path.join(settings.BASE_DIR, 'data'),
    }, name='serve_data_file'),
]
# Thêm middleware CORS nếu cần (khuyến nghị)
if settings.DEBUG:
    from django.conf.urls.static import static
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)