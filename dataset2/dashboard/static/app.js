const ui = {
  connectionDot: document.querySelector('#connectionDot'),
  connectionText: document.querySelector('#connectionText'),
  currentStatus: document.querySelector('#currentStatus'),
  currentRecipe: document.querySelector('#currentRecipe'),
  queueSummary: document.querySelector('#queueSummary'),
  queueBody: document.querySelector('#queueBody'),
  eventCount: document.querySelector('#eventCount'),
  eventsList: document.querySelector('#eventsList'),
  probabilityValue: document.querySelector('#probabilityValue'),
  probabilityBar: document.querySelector('#probabilityBar'),
  thresholdMarker: document.querySelector('#thresholdMarker'),
  modelDetails: document.querySelector('#modelDetails'),
  frameStatus: document.querySelector('#frameStatus'),
  frameShell: document.querySelector('#frameShell'),
  liveFrame: document.querySelector('#liveFrame'),
  framePlaceholder: document.querySelector('#framePlaceholder'),
  frameCaptured: document.querySelector('#frameCaptured'),
  frameAge: document.querySelector('#frameAge'),
  frameSource: document.querySelector('#frameSource'),
  modelRoi: document.querySelector('#modelRoi'),
  ruleRoi: document.querySelector('#ruleRoi'),
  alertBanner: document.querySelector('#alertBanner'),
  modal: document.querySelector('#alertModal'),
  modalCard: document.querySelector('#modalCard'),
  modalEyebrow: document.querySelector('#modalEyebrow'),
  modalTitle: document.querySelector('#modalTitle'),
  modalMessage: document.querySelector('#modalMessage'),
  modalMeta: document.querySelector('#modalMeta'),
  acknowledge: document.querySelector('#acknowledgeButton'),
};

let initialized = false;
let newestEventId = 0;
let pendingAlerts = [];
let activeAlert = null;
let liveFrameVersion = null;

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>'"]/g, character => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'
  })[character]);
}

function formatMl(value) {
  const number = Number(value || 0);
  return `${Number.isInteger(number) ? number : number.toFixed(1)} mL`;
}

function eventTime(value) {
  if (!value) return '—';
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? escapeHtml(value) : date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function riskLevel(event) {
  return String(event?.level || 'RISK').toUpperCase();
}

function positionRoi(element, roi, width, height) {
  if (!Array.isArray(roi) || roi.length !== 4 || !width || !height) {
    element.classList.add('hidden');
    return;
  }
  const [left, top, right, bottom] = roi.map(Number);
  element.classList.remove('hidden');
  element.style.left = `${left / width * 100}%`;
  element.style.top = `${top / height * 100}%`;
  element.style.width = `${(right - left) / width * 100}%`;
  element.style.height = `${(bottom - top) / height * 100}%`;
}

function renderFrame(frame) {
  if (!frame?.available) {
    ui.frameShell.classList.add('empty');
    ui.frameStatus.className = 'camera-status waiting';
    ui.frameStatus.textContent = 'Waiting for frame';
    ui.frameCaptured.textContent = '—';
    ui.frameAge.textContent = '—';
    ui.frameSource.textContent = '—';
    return;
  }
  const age = Math.max(0, Number(frame.age_seconds || 0));
  const fresh = age < 3;
  ui.frameStatus.className = `camera-status ${fresh ? '' : 'stale'}`;
  ui.frameStatus.textContent = fresh ? 'Live frame' : 'Frame delayed';
  ui.frameCaptured.textContent = eventTime(frame.capture_time);
  ui.frameAge.textContent = age < 1 ? '< 1 sec' : `${Math.round(age)} sec`;
  ui.frameSource.textContent = frame.source_name || 'camera';
  const width = Number(frame.width || 640);
  const height = Number(frame.height || 480);
  ui.frameShell.style.aspectRatio = `${width} / ${height}`;
  positionRoi(ui.modelRoi, frame.model_roi, width, height);
  positionRoi(ui.ruleRoi, frame.rule_roi, width, height);
  if (frame.version !== liveFrameVersion) {
    liveFrameVersion = frame.version;
    ui.liveFrame.onload = () => ui.frameShell.classList.remove('empty');
    ui.liveFrame.onerror = () => {
      ui.frameShell.classList.add('empty');
      ui.frameStatus.className = 'camera-status stale';
      ui.frameStatus.textContent = 'Frame unavailable';
    };
    ui.liveFrame.src = `${frame.url}?v=${encodeURIComponent(frame.version)}`;
  }
}

function renderCurrent(current) {
  if (!current) {
    ui.currentRecipe.className = 'current-recipe empty-state';
    ui.currentRecipe.textContent = 'Queue complete or waiting for data.';
    ui.currentStatus.textContent = 'Waiting';
    return;
  }
  const status = current.status === 'filling' ? 'Filling now' : 'Up next';
  ui.currentStatus.textContent = status;
  ui.currentRecipe.className = 'current-recipe';
  ui.currentRecipe.innerHTML = `
    <div class="recipe-title">
      <strong>Recipe ${escapeHtml(current.recipe_id)}</strong>
      <span>Vial <b class="vial-number">${escapeHtml(current.vial)}</b> · ${escapeHtml(current.total_ml)} mL total</span>
    </div>
    <div class="tank-grid">
      <div class="tank clear"><span>T-100 · CLEAR</span><strong>${formatMl(current.t100_ml)}</strong></div>
      <div class="tank orange"><span>T-200 · ORANGE</span><strong>${formatMl(current.t200_ml)}</strong></div>
      <div class="tank blue"><span>T-400 · BLUE</span><strong>${formatMl(current.t400_ml)}</strong></div>
    </div>`;
}

function composition(row) {
  return `100: ${row.t100_ml || 0} · 200: ${row.t200_ml || 0} · 400: ${row.t400_ml || 0}`;
}

function renderQueue(queue, counts) {
  ui.queueSummary.textContent = `${counts.completed}/${counts.total} completed`;
  ui.queueBody.innerHTML = queue.map(row => {
    const current = row.status === 'filling' || row.status === 'next';
    let risk = '<span class="muted">—</span>';
    if (row.defect) risk = `<span class="risk-badge actual">${escapeHtml(row.defect.kind)}</span>`;
    else if (row.prediction) risk = `<span class="risk-badge ${riskLevel(row.prediction).toLowerCase()}">${escapeHtml(riskLevel(row.prediction))}</span>`;
    return `<tr class="${current ? 'current' : escapeHtml(row.status)}">
      <td><strong>${escapeHtml(row.vial)}</strong></td>
      <td>Recipe ${escapeHtml(row.recipe_id)}</td>
      <td class="muted">${escapeHtml(composition(row))}</td>
      <td><span class="queue-status ${escapeHtml(row.status)}">${escapeHtml(row.status)}</span></td>
      <td>${risk}</td>
    </tr>`;
  }).join('');
}

function renderEvents(events) {
  ui.eventCount.textContent = `${events.length} recent`;
  if (!events.length) {
    ui.eventsList.className = 'events-list empty-state';
    ui.eventsList.textContent = 'No monitor events yet.';
    return;
  }
  ui.eventsList.className = 'events-list';
  ui.eventsList.innerHTML = [...events].reverse().map(event => `
    <div class="event">
      <div class="event-head">
        <span class="event-kind ${escapeHtml(event.kind)}">${escapeHtml(event.kind)} · Vial ${escapeHtml(event.vial)}</span>
        <span class="event-time">${eventTime(event.time)}</span>
      </div>
      <p>${escapeHtml(event.message)}</p>
    </div>`).join('');
}

function renderNeural(prediction) {
  if (!prediction) {
    ui.probabilityValue.textContent = '—';
    ui.probabilityBar.style.width = '0%';
    ui.modelDetails.textContent = 'Waiting for neural predictions.';
    return;
  }
  const probability = Math.max(0, Math.min(1, Number(prediction.observed_probability || 0)));
  const threshold = Math.max(0, Math.min(1, Number(prediction.threshold || 0)));
  ui.probabilityValue.textContent = `${(probability * 100).toFixed(1)}%`;
  ui.probabilityBar.style.width = `${probability * 100}%`;
  ui.thresholdMarker.style.left = `${threshold * 100}%`;
  ui.modelDetails.innerHTML = `Vial <strong>${escapeHtml(prediction.vial)}</strong> · ${escapeHtml(prediction.severity_prediction || 'unknown')} severity · ${escapeHtml(prediction.threshold_hits || 0)}/${escapeHtml(prediction.window_frames || 0)} recent frames above threshold · ${Number(prediction.latency_ms || 0).toFixed(1)} ms`;
}

function alertDescription(event) {
  const kind = String(event.kind).toUpperCase();
  if (kind === 'PREDICT') return { type: 'prediction', eyebrow: 'FUTURE RISK', title: 'Future carryover predicted', icon: '↗' };
  if (kind === 'CONFIRM') return { type: 'actual', eyebrow: 'DEFECT CONFIRMED', title: 'Carryover confirmed', icon: '!' };
  if (kind === 'DETECT') return { type: 'actual', eyebrow: 'LIVE DEFECT', title: 'Unexpected color detected', icon: '!' };
  return { type: 'actual', eyebrow: 'NEURAL SHADOW ALERT', title: 'Possible defect observed', icon: '!' };
}

function showNextAlert() {
  if (activeAlert || !pendingAlerts.length) return;
  activeAlert = pendingAlerts.shift();
  const description = alertDescription(activeAlert);
  ui.modalCard.classList.toggle('prediction', description.type === 'prediction');
  ui.modalEyebrow.textContent = description.eyebrow;
  ui.modalTitle.textContent = description.title;
  ui.modalMessage.textContent = activeAlert.message;
  ui.modalMeta.textContent = `Vial ${activeAlert.vial} · ${eventTime(activeAlert.time)} · Operator acknowledgment only`;
  document.querySelector('#modalIcon').textContent = description.icon;
  ui.modal.classList.remove('hidden');
}

function collectAlerts(events) {
  const alertKinds = new Set(['PREDICT', 'MODEL', 'DETECT', 'CONFIRM']);
  const maximumId = Math.max(0, ...events.map(event => Number(event.id || 0)));
  if (!initialized) {
    newestEventId = maximumId;
    initialized = true;
    return;
  }
  // live_monitor.py starts a fresh event log for each run. Notice the line-number
  // reset so the first alerts of a new run are not hidden by the previous run.
  if (maximumId < newestEventId) {
    newestEventId = 0;
    pendingAlerts = [];
  }
  const fresh = events.filter(event => Number(event.id || 0) > newestEventId && alertKinds.has(String(event.kind).toUpperCase()));
  newestEventId = Math.max(newestEventId, maximumId);
  pendingAlerts.push(...fresh);
  showNextAlert();
}

function renderBanner(events) {
  const lastAlert = [...events].reverse().find(event => ['PREDICT', 'MODEL', 'DETECT', 'CONFIRM'].includes(String(event.kind).toUpperCase()));
  if (!lastAlert) { ui.alertBanner.classList.add('hidden'); return; }
  const prediction = String(lastAlert.kind).toUpperCase() === 'PREDICT';
  ui.alertBanner.className = `alert-banner ${prediction ? 'prediction' : ''}`;
  ui.alertBanner.textContent = `${prediction ? 'Future risk' : 'Latest defect notice'} · Vial ${lastAlert.vial}: ${lastAlert.message}`;
}

async function refresh() {
  try {
    const response = await fetch('/api/state', { cache: 'no-store' });
    if (response.status === 401) { window.location.assign('/login'); return; }
    if (!response.ok) throw new Error(`status ${response.status}`);
    const state = await response.json();
    const active = state.monitor.status === 'active';
    ui.connectionDot.className = `status-dot ${active ? '' : 'waiting'}`;
    ui.connectionText.textContent = active ? 'Monitor active' : (state.monitor.has_events ? 'Monitor idle' : 'Waiting for monitor');
    renderCurrent(state.current);
    renderFrame(state.frame);
    renderQueue(state.queue, state.counts);
    renderEvents(state.events);
    renderNeural(state.latest_neural);
    renderBanner(state.events);
    collectAlerts(state.events);
  } catch (error) {
    ui.connectionDot.className = 'status-dot error';
    ui.connectionText.textContent = 'Dashboard disconnected';
  }
}

ui.acknowledge.addEventListener('click', () => {
  ui.modal.classList.add('hidden');
  activeAlert = null;
  showNextAlert();
});

refresh();
setInterval(refresh, 750);
