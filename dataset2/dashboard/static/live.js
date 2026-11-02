const liveUi = {
  body: document.querySelector('#recipeEditorBody'),
  draftStatus: document.querySelector('#draftStatus'),
  queueFeedback: document.querySelector('#queueFeedback'),
  controlFeedback: document.querySelector('#controlFeedback'),
  runState: document.querySelector('#runStateBadge'),
  runDetails: document.querySelector('#runDetails'),
  checks: document.querySelector('#configurationChecks'),
  profile: document.querySelector('#runProfile'),
  startVial: document.querySelector('#startVial'),
  file: document.querySelector('#queueFileInput'),
  add: document.querySelector('#addRecipeRow'),
  import: document.querySelector('#importQueueButton'),
  validate: document.querySelector('#validateQueueButton'),
  save: document.querySelector('#saveQueueButton'),
  start: document.querySelector('#startRunButton'),
  pause: document.querySelector('#pauseRunButton'),
  resume: document.querySelector('#resumeRunButton'),
  realign: document.querySelector('#realignRunButton'),
  stop: document.querySelector('#stopRunButton'),
};

let recipeRows = [];
let csrfToken = '';
let currentControl = {};

function html(value) {
  return String(value ?? '').replace(/[&<>'"]/g, character => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'
  })[character]);
}

function defaultRow() {
  const previous = recipeRows[recipeRows.length - 1];
  return {
    vial: previous ? Number(previous.vial) + 1 : 1,
    recipe_id: previous?.recipe_id ?? 1,
    t100_ml: previous?.t100_ml ?? 10,
    t200_ml: previous?.t200_ml ?? 10,
    t400_ml: previous?.t400_ml ?? 10,
    total_ml: previous?.total_ml ?? 30,
  };
}

function renderEditor() {
  liveUi.body.innerHTML = recipeRows.map((row, index) => `
    <tr data-row="${index}">
      ${['vial', 'recipe_id', 't100_ml', 't200_ml', 't400_ml', 'total_ml'].map(field => `
        <td><input data-field="${field}" type="number" min="0" step="${field.includes('_ml') ? '0.1' : '1'}" value="${html(row[field] ?? '')}" aria-label="${field} row ${index + 1}"></td>
      `).join('')}
      <td><span class="color-preview"><i style="background:${html(row.expected_hex || '#2b3b38')}"></i><b>${html(row.expected_hex || 'auto')}${row.expected_hue !== undefined && row.expected_hue !== '' ? ` · H${html(Math.round(Number(row.expected_hue)))}` : ''}</b></span>${row.transition_risk && row.transition_risk !== '—' ? `<small class="queue-risk ${html(row.transition_risk.toLowerCase())}">${html(row.transition_risk)}</small>` : ''}</td>
      <td><button class="icon-button remove-row" type="button" aria-label="Remove row ${index + 1}">×</button></td>
    </tr>`).join('');
  liveUi.body.querySelectorAll('input').forEach(input => input.addEventListener('input', event => {
    const row = Number(event.target.closest('tr').dataset.row);
    recipeRows[row][event.target.dataset.field] = event.target.value;
    liveUi.draftStatus.textContent = 'Unsaved changes';
  }));
  liveUi.body.querySelectorAll('.remove-row').forEach(button => button.addEventListener('click', event => {
    recipeRows.splice(Number(event.target.closest('tr').dataset.row), 1);
    renderEditor(); updateVialOptions(); liveUi.draftStatus.textContent = 'Unsaved changes';
  }));
  updateVialOptions();
}

function updateVialOptions() {
  const selected = liveUi.startVial.value;
  liveUi.startVial.innerHTML = recipeRows.map(row => `<option value="${html(row.vial)}">Vial ${html(row.vial)}</option>`).join('');
  if ([...liveUi.startVial.options].some(option => option.value === selected)) liveUi.startVial.value = selected;
}

function parseCsv(text) {
  const lines = text.replace(/^\uFEFF/, '').split(/\r?\n/).filter(line => line.trim());
  if (lines.length < 2) throw new Error('CSV needs a header and at least one recipe row.');
  const parseLine = line => {
    const values = []; let value = ''; let quoted = false;
    for (let index = 0; index < line.length; index += 1) {
      const character = line[index];
      if (character === '"' && quoted && line[index + 1] === '"') { value += '"'; index += 1; }
      else if (character === '"') quoted = !quoted;
      else if (character === ',' && !quoted) { values.push(value.trim()); value = ''; }
      else value += character;
    }
    values.push(value.trim()); return values;
  };
  const headers = parseLine(lines[0]);
  const required = ['vial', 'recipe_id', 't100_ml', 't200_ml', 't400_ml'];
  required.forEach(field => { if (!headers.includes(field)) throw new Error(`CSV is missing ${field}.`); });
  return lines.slice(1).map(line => Object.fromEntries(headers.map((header, index) => [header, parseLine(line)[index] ?? ''])));
}

async function api(path, payload) {
  const response = await fetch(path, {
    method: 'POST', cache: 'no-store', headers: {
      'Content-Type': 'application/json', 'X-CSRF-Token': csrfToken,
    }, body: JSON.stringify(payload),
  });
  if (response.status === 401) { window.location.assign('/login'); throw new Error('Authentication required.'); }
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || `Request failed (${response.status}).`);
  return body;
}

function describeControl(control) {
  const status = control.container_status || 'stopped';
  liveUi.runState.textContent = status;
  liveUi.runState.className = `status-chip run-${status}`;
  const details = [];
  if (control.run_id) details.push(`<strong>Run:</strong> ${html(control.run_id)}`);
  if (control.started_at) details.push(`<strong>Started:</strong> ${new Date(control.started_at).toLocaleString()}`);
  if (control.start_vial) details.push(`<strong>Aligned next:</strong> vial ${html(control.start_vial)}`);
  liveUi.runDetails.innerHTML = details.length ? details.join('<br>') : 'No live run has been started.';
  const checks = control.configuration || {};
  liveUi.checks.innerHTML = Object.entries(checks).map(([name, ready]) => `<span class="config-item ${ready ? 'ready' : 'missing'}">${ready ? '✓' : '!'} ${html(name.replace('_', ' '))}</span>`).join('');
  const active = Boolean(control.active);
  liveUi.start.disabled = active || !control.configured || !control.queue_saved;
  liveUi.pause.disabled = status !== 'running';
  liveUi.resume.disabled = status !== 'paused';
  liveUi.stop.disabled = !active;
  liveUi.realign.disabled = !active || !recipeRows.length;
}

async function loadConfig(initial = false) {
  try {
    const response = await fetch('/api/live/config', { cache: 'no-store' });
    if (response.status === 401) { window.location.assign('/login'); return; }
    if (!response.ok) throw new Error(`Configuration unavailable (${response.status}).`);
    const config = await response.json();
    csrfToken = config.csrf_token;
    currentControl = config.control;
    if (initial) {
      recipeRows = config.queue?.length ? config.queue : [defaultRow()];
      renderEditor();
      liveUi.draftStatus.textContent = config.control.queue_saved ? 'Saved queue' : 'Not saved';
    }
    describeControl(config.control);
  } catch (error) {
    liveUi.controlFeedback.textContent = error.message;
    liveUi.controlFeedback.className = 'control-feedback error';
  }
}

async function validateQueue(save = false) {
  liveUi.queueFeedback.textContent = save ? 'Validating and saving…' : 'Validating…';
  liveUi.queueFeedback.className = 'control-feedback';
  try {
    const result = await api(save ? '/api/live/queue' : '/api/live/queue/validate', { rows: recipeRows });
    recipeRows = result.rows;
    renderEditor();
    liveUi.queueFeedback.textContent = `${recipeRows.length} vial${recipeRows.length === 1 ? '' : 's'} validated${save ? ' and saved' : ''}. Expected colors and transition risks are ready.`;
    liveUi.queueFeedback.className = 'control-feedback success';
    liveUi.draftStatus.textContent = save ? 'Saved queue' : 'Validated · not saved';
    if (save) await loadConfig(false);
    return true;
  } catch (error) {
    liveUi.queueFeedback.textContent = error.message;
    liveUi.queueFeedback.className = 'control-feedback error';
    return false;
  }
}

async function control(path, payload, successMessage) {
  liveUi.controlFeedback.textContent = 'Working…';
  liveUi.controlFeedback.className = 'control-feedback';
  try {
    currentControl = await api(path, payload);
    describeControl(currentControl);
    liveUi.controlFeedback.textContent = successMessage;
    liveUi.controlFeedback.className = 'control-feedback success';
  } catch (error) {
    liveUi.controlFeedback.textContent = error.message;
    liveUi.controlFeedback.className = 'control-feedback error';
  }
}

liveUi.add.addEventListener('click', () => { recipeRows.push(defaultRow()); renderEditor(); liveUi.draftStatus.textContent = 'Unsaved changes'; });
liveUi.import.addEventListener('click', () => liveUi.file.click());
liveUi.file.addEventListener('change', async () => {
  try {
    const [file] = liveUi.file.files;
    if (!file) return;
    recipeRows = parseCsv(await file.text()); renderEditor();
    liveUi.draftStatus.textContent = 'Imported · not saved';
    liveUi.queueFeedback.textContent = `${recipeRows.length} rows imported. Validate before saving.`;
  } catch (error) { liveUi.queueFeedback.textContent = error.message; liveUi.queueFeedback.className = 'control-feedback error'; }
  finally { liveUi.file.value = ''; }
});
liveUi.validate.addEventListener('click', () => validateQueue(false));
liveUi.save.addEventListener('click', () => validateQueue(true));
liveUi.start.addEventListener('click', async () => {
  if (!confirm('Start advisory live monitoring? This does not start or control the physical machine.')) return;
  await control('/api/live/start', { profile: liveUi.profile.value, start_vial: Number(liveUi.startVial.value), confirmed: true }, 'Live monitoring started.');
});
liveUi.pause.addEventListener('click', () => control('/api/live/pause', {}, 'Live monitor paused. The physical machine is not affected.'));
liveUi.resume.addEventListener('click', () => control('/api/live/resume', {}, 'Live monitor resumed.'));
liveUi.stop.addEventListener('click', async () => {
  if (!confirm('Stop the advisory monitor? This will not stop the physical machine.')) return;
  await control('/api/live/stop', { confirmed: true }, 'Live monitor stopped. Run outputs were preserved.');
});
liveUi.realign.addEventListener('click', async () => {
  const vial = Number(liveUi.startVial.value);
  if (!confirm(`Realign the monitor so the next detected physical fill is vial ${vial}? The current monitor run will stop and a new timestamped segment will start.`)) return;
  await control('/api/live/realign', { start_vial: vial, confirmed: true }, `Monitor realigned. The next detected fill will be vial ${vial}.`);
});

loadConfig(true);
setInterval(() => loadConfig(false), 1500);
