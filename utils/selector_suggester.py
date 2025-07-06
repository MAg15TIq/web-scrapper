from bs4 import BeautifulSoup
import re

def suggest_selector(html: str, field_name: str) -> str:
    """
    Suggest a CSS selector for a given field name from HTML.
    Tries to match class, id, or text content to the field name.
    """
    soup = BeautifulSoup(html, 'lxml')
    field = field_name.lower().replace('_', ' ')
    # Try to match by class
    for tag in soup.find_all(True, class_=re.compile(field.replace(' ', '-'))):
        class_name = tag.get('class', [None])[0]
        if class_name:
            return f'.{class_name}'
    # Try to match by id
    for tag in soup.find_all(True, id=True):
        if field in tag['id'].replace('-', ' ').replace('_', ' ').lower():
            return f'#{tag["id"]}'
    # Try to match by text content (exact or close)
    for tag in soup.find_all(True):
        if tag.text and field in tag.text.strip().lower():
            if tag.get('class'):
                return f'.{tag.get("class")[0]}'
            elif tag.get('id'):
                return f'#{tag.get("id")}'
            else:
                return tag.name
    # Fallback: return body
    return 'body' 