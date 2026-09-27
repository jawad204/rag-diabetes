// clicking an example fills the box and submits it
document.querySelectorAll('.chip').forEach(chip => {
  chip.addEventListener('click', () => {
    document.getElementById('q').value = chip.textContent;
    document.getElementById('f').requestSubmit();
  });
});

const f = document.getElementById('f');
const q = document.getElementById('q');
const b = document.getElementById('b');
const answer = document.getElementById('answer');
const sources = document.getElementById('sources');

f.addEventListener('submit', async (e) => {
  e.preventDefault();
  b.disabled = true;
  answer.textContent = 'Thinking...';
  answer.className = '';
  sources.innerHTML = '';
  try {
    const res = await fetch('/ask', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question: q.value})
    });
    if (!res.ok) throw new Error('Request failed: ' + res.status);
    const data = await res.json();

    answer.textContent = data.answer;
    if (data.refused) answer.className = 'refused';

    let html = '<div class="meta">relevance ' + data.relevance_score.toFixed(3);
    if (data.sources.length) {
      html += '<br><br><strong>Sources</strong><ul>';
      for (const s of data.sources) {
        html += '<li>[' + s.n + '] <a href="' + s.url + '" target="_blank" rel="noopener">'
             + s.title + '</a></li>';
      }
      html += '</ul>';
    }
    sources.innerHTML = html + '</div>';
  } catch (err) {
    answer.textContent = String(err);
  } finally {
    b.disabled = false;
  }
});
