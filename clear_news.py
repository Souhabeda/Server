# clear_news.py
from db import news_collection

result = news_collection.delete_many({})
print(f"{result.deleted_count} anciens articles supprimés.")
