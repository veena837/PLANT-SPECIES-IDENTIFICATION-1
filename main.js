// image preview + basic client-side validation
const input = document.getElementById('imageInput');
const preview = document.getElementById('preview');
if (input) {
  input.addEventListener('change', () => {
    const f = input.files[0];
    if (!f) return;
    if (!/\.(jpe?g|png)$/i.test(f.name) || f.size > 10 * 1024 * 1024) {
      alert('Please select a JPG/PNG image under 10MB.');
      input.value = ''; preview.style.display = 'none'; return;
    }
    preview.src = URL.createObjectURL(f);
    preview.style.display = 'inline-block';
  });
  document.getElementById('uploadForm').addEventListener('submit', e => {
    if (!input.files.length) { e.preventDefault(); alert('Please choose an image first.'); return; }
    document.getElementById('predictBtn').textContent = '⏳ Predicting...';
  });
}
