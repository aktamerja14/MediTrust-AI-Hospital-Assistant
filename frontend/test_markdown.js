function escapeHtml(text) {
    return text
        .replace(/&/g, "&")
        .replace(/</g, "<")
        .replace(/>/g, ">")
        .replace(/"/g, """)
        .replace(/'/g, "'");
}

function renderMarkdown(text) {
    if (!text) return "";
    let html = escapeHtml(text);

    html = html
      .replace(/^##\s+(.+)$/gm, "<h2>$1</h2>")
      .replace(/^###\s+(.+)$/gm, "<h3>$1</h3>")
      .replace(/^####\s+(.+)$/gm, "<h4>$1</h4>");

    html = html
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/\*(.+?)\*/g, "<em>$1</em>");

    html = html
      .replace(/^\-\s+(.+)$/gm, "<li>$1</li>");

    html = html
      .replace(/(<li>.*<\/li>)/s, "<ul>$1</ul>");

    html = html
      .replace(/\n{2,}/g, "</p><p>")
      .replace(/\n/g, "<br>");

    html = "<p>" + html + "</p>";

    html = html
      .replace(/<p><\/p>/g, "")
      .replace(/<p><h2>/g, "<h2>")
      .replace(/<\/h2><\/p>/g, "</h2>")
      .replace(/<p><h3>/g, "<h3>")
      .replace(/<\/h3><\/p>/g, "</h3>")
      .replace(/<p><h4>/g, "<h4>")
      .replace(/<\/h4><\/p>/g, "</h4>")
      .replace(/<p><ul>/g, "<ul>")
      .replace(/<\/ul><\/p>/g, "</ul>")
      .replace(/<p><li>/g, "<li>")
      .replace(/<\/li><\/p>/g, "</li>")
      .replace(/<p><br>/g, "<p>")
      .replace(/<br><\/p>/g, "</p>");

    return html;
}

const testCases = [
    "Simple paragraph",
    "## Heading 2\n\nParagraph with **bold** and *italic* text.",
    "## Common Symptoms\n- Symptom 1\n- Symptom 2\n- Symptom 3",
    "## What is it?\n- **Definition:** A condition\n- **Types:** Type 1, Type 2\n\n## Treatment Options\n- Medication\n- Lifestyle changes",
];

testCases.forEach((tc, i) => {
    console.log("=== Test", i+1, "===");
    console.log("Input:", JSON.stringify(tc));
    console.log("Output:", renderMarkdown(tc));
    console.log();
});