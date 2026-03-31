Project-LLMs-Rag-Agent/
│
├── manage.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── rag_django/          ← project Django
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   ├── asgi.py
│   └── wsgi.py
│
├── rag_app/             ← app Django
│   ├── templates/
│   │    └── index.html
│   │
│   ├── static/
│   │    └── styles/
│   │         └── style.css
│   │
│   ├── views.py
│   ├── urls.py
│   └── apps.py
│
├── rag/                 ← GIỮ NGUYÊN
│   ├── pipeline.py
│   ├── embedding.py
│   ├── retriever.py
│   ├── promt.py
│   └── llm.py
│
├── document_processing/ ← GIỮ NGUYÊN
│   └── pdf_loader.py
│
├── vector_store/        ← GIỮ NGUYÊN
│   └── faiss_store.py
│
├── data/
│
└── documentation/
