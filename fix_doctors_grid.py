with open(r'F:\AI & ML\Gen AI\MediTrust\frontend\index.html', 'r', encoding='utf-8') as f:
    content = f.read()

old = """<!-- Doctors Cards Grid -->
        <div class="doctors-grid" id="doctors-grid">
          <!-- Doctors will be loaded dynamically from API -->
          <div class="doctor-card" data-specialty="cardiology">"""

new = """<!-- Doctors Cards Grid -->
        <div class="doctors-grid" id="doctors-grid">
          <!-- Doctors will be loaded dynamically from API -->"""

content = content.replace(old, new)
print('First replacement done')

# Now find and remove the rest up to the closing </div> of doctors-grid
end_marker = '        </div>\n      </div>\n    </section>'
end_idx = content.find(end_marker)
if end_idx != -1:
    start_idx = content.find('<!-- Doctor 2: Neurology -->')
    if start_idx != -1:
        content = content[:start_idx] + content[end_idx:]
        print('Second replacement done')

with open(r'F:\AI & ML\Gen AI\MediTrust\frontend\index.html', 'w', encoding='utf-8') as f:
    f.write(content)
print('Done')