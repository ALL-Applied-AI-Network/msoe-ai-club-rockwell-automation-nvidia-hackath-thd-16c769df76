const demoUi = {
  state: document.querySelector('#demoRunState'),
  details: document.querySelector('#demoRunDetails'),
  checks: document.querySelector('#demoConfiguration'),
  feedback: document.querySelector('#demoFeedback'),
  start: document.querySelector('#demoStartButton'),
  pause: document.querySelector('#demoPauseButton'),
  resume: document.querySelector('#demoResumeButton'),
  restart: document.querySelector('#demoRestartButton'),
  stop: document.querySelector('#demoStopButton'),
};

let demoCsrf = '';

function demoEscape(value) {
  return String(value ?? '').replace(/[&<>'"]/g, character => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'
  })[character]);
}

function describeDemo(control) {
  const status = control.container_status || 'stopped';
  demoUi.state.textContent = status;
  demoUi.state.className = `status-chip run-${status}`;
  demoUi.details.textContent = control.run_id
    ? `Run ${control.run_id} · 1× timing · approximately ${control.estimated_minutes} minutes`
    : 'Ready to create a fresh model run from Dataset 5.';
  demoUi.checks.innerHTML = Object.entries(control.configuration || {}).map(([name, ready]) =>
    `<span class="config-item ${ready ? 'ready' : 'missing'}">${ready ? '✓' : '!'} ${demoEscape(name.replaceAll('_', ' '))}</span>`
  ).join('');
  const active = Boolean(control.active);
  demoUi.start.disabled = active || !control.configured;
  demoUi.pause.disabled = status !== 'running';
  demoUi.resume.disabled = status !== 'paused';
  demoUi.restart.disabled = !control.configured;
  demoUi.stop.disabled = !active;
}

async function demoApi(path, confirmed = false) {
  const response = await fetch(path, {
    method: 'POST', cache: 'no-store',
    headers: {'Content-Type': 'application/json', 'X-CSRF-Token': demoCsrf},
    body: JSON.stringify({confirmed}),
  });
  if (response.status === 401) { window.location.assign('/login'); throw new Error('Authentication required.'); }
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || `Request failed (${response.status}).`);
  return body;
}

async function loadDemoConfig() {
  try {
    const response = await fetch('/api/demo/config', {cache: 'no-store'});
    if (response.status === 401) { window.location.assign('/login'); return; }
    if (!response.ok) throw new Error(`Demo configuration unavailable (${response.status}).`);
    const config = await response.json();
    demoCsrf = config.csrf_token;
    describeDemo(config.control);
  } catch (error) {
    demoUi.feedback.textContent = error.message;
    demoUi.feedback.className = 'control-feedback error';
  }
}

async function demoControl(path, message, confirmed = false) {
  demoUi.feedback.textContent = 'Working…';
  demoUi.feedback.className = 'control-feedback';
  try {
    describeDemo(await demoApi(path, confirmed));
    demoUi.feedback.textContent = message;
    demoUi.feedback.className = 'control-feedback success';
  } catch (error) {
    demoUi.feedback.textContent = error.message;
    demoUi.feedback.className = 'control-feedback error';
  }
}

demoUi.start.addEventListener('click', async () => {
  if (!confirm('Start a fresh approximately 28-minute Dataset 5 replay through the frozen model?')) return;
  await demoControl('/api/demo/start', 'Dataset 5 model demonstration started.', true);
});
demoUi.pause.addEventListener('click', () => demoControl('/api/demo/pause', 'Demonstration paused.'));
demoUi.resume.addEventListener('click', () => demoControl('/api/demo/resume', 'Demonstration resumed.'));
demoUi.restart.addEventListener('click', async () => {
  if (!confirm('Restart from the beginning? The current demonstration output will remain preserved.')) return;
  await demoControl('/api/demo/restart', 'A fresh Dataset 5 demonstration started from the beginning.', true);
});
demoUi.stop.addEventListener('click', async () => {
  if (!confirm('Stop the current Dataset 5 demonstration? Results collected so far will remain preserved.')) return;
  await demoControl('/api/demo/stop', 'Demonstration stopped; its partial results were preserved.', true);
});

loadDemoConfig();
setInterval(loadDemoConfig, 1500);
