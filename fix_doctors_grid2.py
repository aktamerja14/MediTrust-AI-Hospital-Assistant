with open(r'F:\AI & ML\Gen AI\MediTrust\frontend\index.html', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the start and end of the doctors section
start_marker = '<!-- Doctors Cards Grid -->'
end_marker = '        </div>\n      </div>\n    </section>'

start_idx = content.find(start_marker)
end_idx = content.find(end_marker, start_idx)

if start_idx != -1 and end_idx != -1:
    # Include the end marker
    end_idx += len(end_marker)
    
    # Replacement content
    new_content = """<!-- Doctors Cards Grid -->
        <div class="doctors-grid" id="doctors-grid">
          <!-- Doctors will be loaded dynamically from API -->
        </div>
      </div>
    </section>"""
    
    content = content[:start_idx] + new_content + content[end_idx:]
    print('Replacement done')
else:
    print('Markers not found')
    print(f'start_idx: {start_idx}')
    print(f'end_idx: {end_idx}')

with open(r'F:\AI & ML\Gen AI\MediTrust\frontend\index.html', 'w', encoding='utf-8') as f:
    f.write(content)
print('Done')