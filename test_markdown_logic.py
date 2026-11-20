import re

def render_markdown(text):
    if not text:
        return ''
    html = text
    
    # Headings
    html = re.sub(r'^##\s+(.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^###\s+(.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    html = re.sub(r'^####\s+(.+)$', r'<h4>\1</h4>', html, flags=re.MULTILINE)
    
    # Bold and italic
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
    html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)
    
    # Bullet lists - convert - items to <li>
    html = re.sub(r'^\-\s+(.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)
    
    # Split by double newlines into blocks FIRST
    blocks = re.split(r'\n{2,}', html)
    html_blocks = []
    
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        
        # Check if block STARTS with a heading followed by list items
        heading_list_match = re.match(r'^(<h[2-4]>.*?</h[2-4]>)\n((?:<li>.*?</li>\n?)+)$', block, re.DOTALL)
        if heading_list_match:
            heading = heading_list_match.group(1)
            list_content = heading_list_match.group(2)
            li_matches = re.findall(r'<li>.*?</li>', list_content, flags=re.DOTALL)
            if li_matches:
                list_html = '<ul>' + ''.join(li_matches) + '</ul>'
                html_blocks.append(heading + '\n' + list_html)
            else:
                html_blocks.append(block)
            continue
        
        # Check if block contains only list items
        lines = block.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        
        is_list_block = all(l.strip().startswith('<li>') for l in non_empty_lines)
        
        if is_list_block:
            li_matches = re.findall(r'<li>.*?</li>', block, flags=re.DOTALL)
            if li_matches:
                block = '<ul>' + ''.join(li_matches) + '</ul>'
            html_blocks.append(block)
        elif (block.startswith('<h2>') or block.startswith('<h3>') or block.startswith('<h4>') or 
              block.startswith('<ul>') or block.startswith('<ol>')):
            html_blocks.append(block)
        else:
            block = block.replace('\n', '<br>')
            html_blocks.append('<p>' + block + '</p>')
    
    html = '\n'.join(html_blocks)
    
    # Clean up
    html = html.replace('<p></p>', '')
    html = html.replace('<p><br>', '<p>')
    html = html.replace('<br></p>', '</p>')
    
    return html

# Test content with single newlines in paragraph
test_content = """First paragraph line one.
First paragraph line two.

## Heading
- Item 1
- Item 2

Second paragraph line one.
Second paragraph line two."""

result = render_markdown(test_content)
print("=== RENDERED HTML ===")
print(result)
print("\n=== CHECKING ===")
print("Has br:", '<br>' in result)