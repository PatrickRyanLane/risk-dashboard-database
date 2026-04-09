import { mountParityPage, prefetchParityPage } from '../../parity-runtime.js';

const page = {
  "styles": ":root{\n  --bg:#044152;\n  --card:#092e37;\n  --cardAlt:#0d3944;\n  --muted:#a2ebf3;\n  --text:#ebf2f2;\n  --accent:#58dbed;\n  --accentSoft:rgba(88,219,237,.18);\n  --stroke:#0e2230;\n  --chip:#12343e;\n  --chipBorder:#174550;\n  --ink:#1f2121;\n  --danger:#ff8261;\n  --dangerBorder:#7b1d1d;\n}\n*{box-sizing:border-box}\nbody{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 Inter,system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif}\n.wrap{max-width:1440px;margin:0 auto;padding:20px}\nh1,h2,h3{margin:0}\np{margin:0}\n.muted{color:var(--muted)}\n.card{background:var(--card);border:1px solid var(--stroke);border-radius:14px;padding:16px}\n.page-head{display:flex;align-items:flex-end;justify-content:space-between;gap:16px;margin-bottom:16px}\n.page-head p{max-width:860px}\n.controls{display:flex;flex-direction:column;gap:12px;margin-bottom:16px}\n.controls-grid{display:grid;grid-template-columns:220px minmax(260px,420px) auto;gap:12px;align-items:end}\n.field-label{font-size:12px;padding-left:8px;color:var(--muted);display:block;margin-bottom:6px}\nselect,button{height:36px;border-radius:10px;border:1px solid #23314d;background:var(--ink);color:var(--text);padding:8px 12px;font:inherit}\nbutton{cursor:pointer}\nbutton:hover{filter:brightness(1.08)}\n.lookback-controls{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px}\n.lookback-btn{font-weight:600}\n.lookback-btn.active{border-color:var(--accent);box-shadow:0 0 0 1px rgba(88,219,237,.35);color:var(--accent)}\n.status-line{font-size:12px;min-height:18px}\n.status-line.error{color:var(--danger)}\n.stats-grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;margin-bottom:16px}\n.stat-card{background:var(--cardAlt);border:1px solid var(--stroke);border-radius:12px;padding:14px}\n.stat-card .label{display:block;font-size:12px;color:var(--muted);margin-bottom:8px}\n.stat-card strong{font-size:24px;line-height:1}\n.stack{display:flex;flex-direction:column;gap:16px}\n.section-head{display:flex;align-items:flex-start;justify-content:space-between;gap:12px;margin-bottom:12px}\n.section-head p{max-width:720px}\n.detail-controls{min-width:260px}\n.detail-controls select{width:100%}\n.subsection-head{display:flex;align-items:flex-start;justify-content:space-between;gap:12px;margin:18px 0 10px}\n.table-scroll{overflow:auto;-webkit-overflow-scrolling:touch}\n.table-scroll::-webkit-scrollbar{height:8px;width:8px}\n.table-scroll::-webkit-scrollbar-track{background:rgba(255,255,255,.05);border-radius:999px}\n.table-scroll::-webkit-scrollbar-thumb{background:rgba(255,255,255,.16);border-radius:999px}\ntable{width:100%;border-collapse:separate;border-spacing:0 8px}\nthead th{font-size:11px;font-weight:600;color:var(--muted);text-align:left;padding:0 10px 4px;vertical-align:bottom;white-space:nowrap}\nthead th.numeric{text-align:right}\ntbody td{background:var(--ink);border-top:1px solid #0f1831;border-bottom:1px solid #0f1831;padding:10px;vertical-align:top}\ntbody td.numeric{text-align:right;font-variant-numeric:tabular-nums}\ntbody tr td:first-child{border-left:1px solid #0f1831;border-radius:12px 0 0 12px}\ntbody tr td:last-child{border-right:1px solid #0f1831;border-radius:0 12px 12px 0}\n.table-empty td{text-align:center;color:var(--muted);padding:18px 12px}\n.tag-pill{display:inline-flex;align-items:center;gap:6px;padding:3px 10px;border-radius:999px;border:1px solid var(--dangerBorder);background:rgba(255,130,97,.08);color:#ffd7cd;font-size:12px;font-weight:600}\n.crisis-row{cursor:pointer}\n.crisis-row:hover td{background:#253640}\n.crisis-row.selected td{background:rgba(88,219,237,.12)}\n.crisis-row.selected td:first-child{border-left-color:var(--accent)}\n.crisis-row.selected td:last-child{border-right-color:var(--accent)}\n.detail-shell{display:flex;flex-direction:column;gap:14px}\n.detail-empty{padding:28px 20px;border:1px dashed rgba(162,235,243,.28);border-radius:12px;color:var(--muted);text-align:center}\n.detail-summary{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}\n.detail-mini{background:var(--cardAlt);border:1px solid var(--stroke);border-radius:12px;padding:12px}\n.detail-mini .label{display:block;font-size:12px;color:var(--muted);margin-bottom:6px}\n.detail-mini strong{font-size:20px;line-height:1}\n.table-block{display:flex;flex-direction:column;gap:10px}\n.table-block h3{font-size:15px}\n.brand-name{font-weight:600}\n.yes-pill,.no-pill{display:inline-flex;align-items:center;padding:3px 8px;border-radius:999px;font-size:12px;border:1px solid transparent}\n.yes-pill{background:rgba(88,219,237,.14);border-color:rgba(88,219,237,.35);color:var(--accent)}\n.no-pill{background:rgba(255,255,255,.05);border-color:rgba(255,255,255,.12);color:var(--muted)}\n.trend-chart-shell{position:relative;min-height:340px;height:42vh;max-height:520px}\n.trend-chart-shell canvas{width:100% !important;height:100% !important;display:block}\n.chart-empty{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;border:1px dashed rgba(162,235,243,.28);border-radius:12px;color:var(--muted);background:rgba(0,0,0,.08)}\n@media (max-width: 900px){\n  .controls-grid{grid-template-columns:1fr}\n  .stats-grid{grid-template-columns:repeat(2,minmax(0,1fr))}\n  .detail-summary{grid-template-columns:repeat(2,minmax(0,1fr))}\n  .section-head,.subsection-head{flex-direction:column}\n  .detail-controls{min-width:0;width:100%}\n}\n@media (max-width: 620px){\n  .wrap{padding:16px}\n  .stats-grid{grid-template-columns:1fr}\n  .detail-summary{grid-template-columns:1fr}\n}",
  "markup": "<div class=\"wrap\">\n  <header class=\"page-head\">\n    <div>\n      <h1>Crisis Dashboard: Crises</h1>\n      <p class=\"muted\">Track which crisis categories are expanding, which are still active, and which brands are carrying each category in the selected window.</p>\n    </div>\n  </header>\n\n  <section class=\"card controls\">\n    <div class=\"controls-grid\">\n      <div>\n        <label class=\"field-label\" for=\"dateSelect\">Window End Date</label>\n        <select id=\"dateSelect\"></select>\n      </div>\n      <div>\n        <span class=\"field-label\">Lookback Window</span>\n        <div class=\"lookback-controls\" role=\"group\" aria-label=\"Lookback window\">\n          <button type=\"button\" class=\"lookback-btn\" data-days=\"30\">30d</button>\n          <button type=\"button\" class=\"lookback-btn\" data-days=\"45\">45d</button>\n          <button type=\"button\" class=\"lookback-btn\" data-days=\"60\">60d</button>\n          <button type=\"button\" class=\"lookback-btn\" data-days=\"90\">90d</button>\n        </div>\n      </div>\n      <div class=\"muted\">\n        Crisis rows come from tagged daily crisis events and recent negative coverage in the selected window.\n      </div>\n    </div>\n    <div id=\"statusLine\" class=\"status-line muted\"></div>\n  </section>\n\n  <section class=\"stats-grid\">\n    <div class=\"stat-card\">\n      <span class=\"label\">Crises In Window</span>\n      <strong id=\"statCrisisCount\">0</strong>\n    </div>\n    <div class=\"stat-card\">\n      <span class=\"label\">Distinct Brands Affected</span>\n      <strong id=\"statBrandCount\">0</strong>\n    </div>\n    <div class=\"stat-card\">\n      <span class=\"label\">Active Brands</span>\n      <strong id=\"statActiveBrandCount\">0</strong>\n    </div>\n    <div class=\"stat-card\">\n      <span class=\"label\">Average Crisis Length</span>\n      <strong id=\"statAvgCrisisLength\">0d</strong>\n    </div>\n    <div class=\"stat-card\">\n      <span class=\"label\">Brand-Days</span>\n      <strong id=\"statBrandDays\">0</strong>\n    </div>\n  </section>\n\n  <section class=\"stack\">\n    <div class=\"card\">\n      <div class=\"section-head\">\n        <div>\n          <h2>Crisis Summary</h2>\n          <p class=\"muted\">Click a crisis category to update the selector and focus the detail panel below.</p>\n        </div>\n      </div>\n      <div class=\"table-scroll\">\n        <table>\n          <thead>\n            <tr>\n              <th>Crisis Type</th>\n              <th class=\"numeric\">Affected Brands</th>\n              <th class=\"numeric\">Active Brands</th>\n              <th class=\"numeric\">Average Active Days</th>\n              <th class=\"numeric\">Longest Active Days</th>\n            </tr>\n          </thead>\n          <tbody id=\"summaryBody\"></tbody>\n        </table>\n      </div>\n    </div>\n\n    <div class=\"card\">\n      <div class=\"section-head\">\n        <div>\n          <h2>Crisis Trends</h2>\n          <p class=\"muted\">Daily active-brand counts for the most visible crisis categories in the selected window. The chart highlights the leading categories; the table shows the latest dates in the same window.</p>\n        </div>\n      </div>\n      <div class=\"trend-chart-shell\">\n        <canvas id=\"trendChart\"></canvas>\n        <div id=\"trendChartEmpty\" class=\"chart-empty\" hidden>Trend data will appear here when crisis categories are available.</div>\n      </div>\n\n      <div class=\"subsection-head\">\n        <div>\n          <h3>Trend Table</h3>\n          <p id=\"trendTableNote\" class=\"muted\"></p>\n        </div>\n      </div>\n      <div class=\"table-scroll\">\n        <table>\n          <thead id=\"trendHead\"></thead>\n          <tbody id=\"trendBody\"></tbody>\n        </table>\n      </div>\n    </div>\n\n    <div class=\"card\">\n      <div class=\"section-head\">\n        <div>\n          <h2 id=\"detailHeading\">Selected Crisis</h2>\n          <p id=\"detailSubhead\" class=\"muted\">Choose a crisis from the summary table to see the brands carrying it in the current window.</p>\n        </div>\n        <div class=\"detail-controls\">\n          <label class=\"field-label\" for=\"crisisSelect\">Selected Crisis</label>\n          <select id=\"crisisSelect\" disabled>\n            <option value=\"\">Select a crisis</option>\n          </select>\n        </div>\n      </div>\n\n      <div id=\"detailEmpty\" class=\"detail-empty\">No crisis is selected for this window yet.</div>\n\n      <div id=\"detailContent\" class=\"detail-shell\" hidden>\n        <div class=\"detail-summary\">\n          <div class=\"detail-mini\">\n            <span class=\"label\">Brands Affected</span>\n            <strong id=\"detailBrandCount\">0</strong>\n          </div>\n          <div class=\"detail-mini\">\n            <span class=\"label\">Active Brands</span>\n            <strong id=\"detailActiveCount\">0</strong>\n          </div>\n          <div class=\"detail-mini\">\n            <span class=\"label\">Average Active Days</span>\n            <strong id=\"detailAvgActiveDays\">0d</strong>\n          </div>\n          <div class=\"detail-mini\">\n            <span class=\"label\">Longest Active Days</span>\n            <strong id=\"detailLongestActiveDays\">0d</strong>\n          </div>\n        </div>\n\n        <div class=\"table-block\">\n          <h3>Affected Brands</h3>\n          <div class=\"table-scroll\">\n            <table>\n              <thead>\n                <tr>\n                  <th>Brand</th>\n                  <th>Sector</th>\n                  <th class=\"numeric\">Active Days</th>\n                  <th>First Seen</th>\n                  <th>Last Seen</th>\n                  <th>Still Active?</th>\n                </tr>\n              </thead>\n              <tbody id=\"brandBody\"></tbody>\n            </table>\n          </div>\n        </div>\n      </div>\n    </div>\n  </section>\n</div>",
  "externalScripts": [
    "https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js",
    "/static/app/shared-data.js"
  ]
};

function runInlineScript(scope) {
  const {
    window,
    self,
    top,
    parent,
    document,
    history,
    location,
    globalThis,
    fetch,
    localStorage,
    sessionStorage,
    navigator,
    requestAnimationFrame,
    cancelAnimationFrame,
    requestIdleCallback,
    cancelIdleCallback,
    setTimeout,
    clearTimeout,
    setInterval,
    clearInterval,
    performance,
    console,
    URL,
    URLSearchParams,
    Chart,
    Papa,
    ResizeObserver,
    MutationObserver,
    Event,
    CustomEvent,
    Node,
    HTMLElement,
    HTMLCanvasElement,
    CSS,
    getComputedStyle
  } = scope;

  const LOOKBACK_OPTIONS = [30, 45, 60, 90];
  const TREND_COLORS = ['#ff3b30', '#ff6b57', '#f59e0b', '#facc15', '#d1d5db', '#bfdbfe', '#22c55e', '#a78bfa'];
  const MAX_TREND_SERIES = 6;
  const TREND_TABLE_DAYS = 7;

  const params = new URLSearchParams(window.location.search);
  const requestedDays = Number(params.get('days') || '');
  let currentDays = LOOKBACK_OPTIONS.includes(requestedDays) ? requestedDays : 30;
  let selectedCrisisTag = (params.get('crisis_tag') || '').trim();
  let availableDates = [];
  let activeRequestToken = 0;
  let trendChart = null;

  const esc = (value) => String(value ?? '').replace(/[&<>\"']/g, (match) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  }[match]));

  function setStatus(message, isError = false){
    const node = document.getElementById('statusLine');
    node.textContent = message || '';
    node.classList.toggle('error', !!isError);
  }

  function syncLookbackButtons(){
    document.querySelectorAll('.lookback-btn').forEach((button) => {
      button.classList.toggle('active', Number(button.dataset.days) === currentDays);
    });
  }

  function buildStartDate(endIso, days){
    const end = new Date(`${endIso}T00:00:00Z`);
    end.setUTCDate(end.getUTCDate() - (days - 1));
    return end.toISOString().slice(0, 10);
  }

  const _jsonCache = new Map();
  async function fetchJson(url){
    const sharedData = window.CrisisDashboardData;
    if (sharedData?.fetchJson) {
      return sharedData.fetchJson(url);
    }
    if (_jsonCache.has(url)) {
      const cached = _jsonCache.get(url);
      return cached instanceof Promise ? await cached : cached;
    }
    const pending = (async () => {
      const response = await fetch(url, { cache: 'default' });
      if (!response.ok){
        let detail = `HTTP ${response.status}`;
        try {
          const payload = await response.json();
          if (payload && payload.error) detail = payload.error;
        } catch (_err) {
          // Ignore JSON parse failures for error bodies.
        }
        throw new Error(detail);
      }
      return await response.json();
    })();
    _jsonCache.set(url, pending);
    try {
      const data = await pending;
      _jsonCache.set(url, data);
      return data;
    } catch (error) {
      _jsonCache.delete(url);
      throw error;
    }
  }

  function renderActivePill(isActive){
    return isActive
      ? '<span class="yes-pill">Active</span>'
      : '<span class="no-pill">Inactive</span>';
  }

  function fmtDays(value){
    const num = Number(value || 0);
    if (!Number.isFinite(num) || num <= 0) return '0d';
    const rounded = Math.round(num * 10) / 10;
    return `${Number.isInteger(rounded) ? rounded.toFixed(0) : rounded.toFixed(1)}d`;
  }

  function shortDateLabel(isoDate, includeYear = false){
    const parts = String(isoDate || '').split('-');
    if (parts.length !== 3) return String(isoDate || '');
    const month = Number(parts[1]);
    const day = Number(parts[2]);
    if (includeYear) return `${month}/${day}/${parts[0].slice(2)}`;
    return `${month}/${day}`;
  }

  function updateSummaryStats(payload){
    const rows = Array.isArray(payload.crises) ? payload.crises : [];
    const avgCrisisLength = rows.length
      ? rows.reduce((sum, row) => sum + Number(row.avg_active_days || 0), 0) / rows.length
      : 0;
    document.getElementById('statCrisisCount').textContent = String(payload.crisis_count || 0);
    document.getElementById('statBrandCount').textContent = String(payload.affected_brand_count || 0);
    document.getElementById('statActiveBrandCount').textContent = String(payload.active_brand_count || 0);
    document.getElementById('statAvgCrisisLength').textContent = fmtDays(avgCrisisLength);
    document.getElementById('statBrandDays').textContent = String(payload.brand_day_count || 0);
  }

  function renderSummaryTable(payload){
    const tbody = document.getElementById('summaryBody');
    const rows = Array.isArray(payload.crises) ? payload.crises : [];
    if (!rows.length){
      tbody.innerHTML = '<tr class="table-empty"><td colspan="5">No crisis tags were active in this window.</td></tr>';
      return;
    }

    const selectedKey = ((payload.selected_crisis && payload.selected_crisis.tag) || selectedCrisisTag || '').toLowerCase();
    tbody.innerHTML = rows.map((row) => {
      const isSelected = String(row.tag || '').toLowerCase() === selectedKey;
      return `<tr class="crisis-row${isSelected ? ' selected' : ''}" data-crisis-tag="${esc(row.tag)}">
        <td><span class="tag-pill">${esc(row.display_tag || row.tag || '')}</span></td>
        <td class="numeric">${row.brands_affected || 0}</td>
        <td class="numeric">${row.active_brands_latest || 0}</td>
        <td class="numeric">${fmtDays(row.avg_active_days || 0)}</td>
        <td class="numeric">${fmtDays(row.longest_active_days || 0)}</td>
      </tr>`;
    }).join('');

    tbody.querySelectorAll('[data-crisis-tag]').forEach((row) => {
      row.addEventListener('click', () => {
        selectedCrisisTag = String(row.getAttribute('data-crisis-tag') || '').trim();
        loadCrisisWindow();
      });
    });
  }

  function renderCrisisSelect(payload){
    const select = document.getElementById('crisisSelect');
    const rows = Array.isArray(payload.crises) ? payload.crises : [];
    const selectedTag = (payload.selected_crisis && payload.selected_crisis.tag) || selectedCrisisTag || '';

    if (!rows.length){
      select.innerHTML = '<option value="">Select a crisis</option>';
      select.disabled = true;
      return;
    }

    select.disabled = false;
    select.innerHTML = rows.map((row) => (
      `<option value="${esc(row.tag)}">${esc(row.display_tag || row.tag || '')}</option>`
    )).join('');
    if (selectedTag) select.value = selectedTag;
  }

  function renderSelectedCrisis(payload){
    const detailHeading = document.getElementById('detailHeading');
    const detailSubhead = document.getElementById('detailSubhead');
    const detailEmpty = document.getElementById('detailEmpty');
    const detailContent = document.getElementById('detailContent');
    const selected = payload.selected_crisis;

    renderCrisisSelect(payload);

    if (!selected){
      detailHeading.textContent = 'Selected Crisis';
      detailSubhead.textContent = 'Choose a crisis from the summary table to see the brands carrying it in the current window.';
      detailEmpty.hidden = false;
      detailContent.hidden = true;
      document.getElementById('brandBody').innerHTML = '';
      return;
    }

    detailEmpty.hidden = true;
    detailContent.hidden = false;
    detailHeading.textContent = selected.display_tag || selected.tag || 'Selected crisis';
    detailSubhead.textContent = `Visible on ${selected.crisis_days || 0} dates in the selected window, with ${selected.total_negative_items || 0} tagged negative items behind the crisis signal.`;
    document.getElementById('detailBrandCount').textContent = String(selected.brands_affected || 0);
    document.getElementById('detailActiveCount').textContent = String(selected.active_brands_latest || 0);
    document.getElementById('detailAvgActiveDays').textContent = fmtDays(selected.avg_active_days || 0);
    document.getElementById('detailLongestActiveDays').textContent = fmtDays(selected.longest_active_days || 0);

    const brandBody = document.getElementById('brandBody');
    const brands = Array.isArray(selected.brands) ? selected.brands : [];
    if (!brands.length){
      brandBody.innerHTML = '<tr class="table-empty"><td colspan="6">No brands were attached to this crisis in the selected window.</td></tr>';
      return;
    }

    brandBody.innerHTML = brands.map((brand) => `<tr>
      <td><span class="brand-name">${esc(brand.brand || '')}</span></td>
      <td>${esc(brand.sector || 'Unassigned')}</td>
      <td class="numeric">${brand.days_affected || 0}</td>
      <td>${esc(brand.first_seen_date || '')}</td>
      <td>${esc(brand.last_seen_date || '')}</td>
      <td>${renderActivePill(!!brand.active_on_window_end)}</td>
    </tr>`).join('');
  }

  function selectTrendRowsForChart(payload){
    const rows = Array.isArray(payload.trend_rows) ? payload.trend_rows : [];
    const selectedKey = ((payload.selected_crisis && payload.selected_crisis.tag) || selectedCrisisTag || '').toLowerCase();
    const selectedRow = rows.find((row) => String(row.tag || '').toLowerCase() === selectedKey);
    let visibleRows = rows.slice(0, MAX_TREND_SERIES);
    if (selectedRow && !visibleRows.some((row) => String(row.tag || '').toLowerCase() === selectedKey)){
      visibleRows = visibleRows.slice(0, Math.max(0, MAX_TREND_SERIES - 1));
      visibleRows.push(selectedRow);
    }
    return visibleRows;
  }

  function destroyTrendChart(){
    if (trendChart){
      trendChart.destroy();
      trendChart = null;
    }
  }

  function renderTrendChart(payload){
    const dates = Array.isArray(payload.trend_dates) ? payload.trend_dates : [];
    const rows = selectTrendRowsForChart(payload);
    const canvas = document.getElementById('trendChart');
    const empty = document.getElementById('trendChartEmpty');

    destroyTrendChart();

    if (!dates.length || !rows.length){
      canvas.hidden = true;
      empty.hidden = false;
      return;
    }

    canvas.hidden = false;
    empty.hidden = true;

    trendChart = new Chart(canvas.getContext('2d'), {
      type: 'line',
      data: {
        labels: dates.map((dateValue) => shortDateLabel(dateValue)),
        datasets: rows.map((row, index) => {
          const color = TREND_COLORS[index % TREND_COLORS.length];
          return {
            label: row.display_tag || row.tag || `Crisis ${index + 1}`,
            data: Array.isArray(row.active_brands_series) ? row.active_brands_series : [],
            borderColor: color,
            backgroundColor: color,
            pointRadius: 0,
            pointHoverRadius: 4,
            borderWidth: 3,
            tension: 0.24,
          };
        }),
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          mode: 'index',
          intersect: false,
        },
        plugins: {
          legend: {
            position: 'right',
            labels: {
              color: '#ebf2f2',
              boxWidth: 14,
              boxHeight: 3,
              usePointStyle: false,
            },
          },
          tooltip: {
            callbacks: {
              title: (items) => {
                const idx = items?.[0]?.dataIndex ?? 0;
                return dates[idx] || '';
              },
              label: (context) => {
                const label = context.dataset?.label || 'Series';
                return `${label}: ${context.parsed?.y ?? 0}`;
              },
            },
          },
        },
        scales: {
          x: {
            ticks: {
              color: '#ebf2f2',
              maxTicksLimit: 10,
            },
            grid: {
              color: 'rgba(255,255,255,.08)',
            },
          },
          y: {
            beginAtZero: true,
            ticks: {
              color: '#ebf2f2',
            },
            grid: {
              color: 'rgba(255,255,255,.08)',
            },
          },
        },
      },
    });
  }

  function renderTrendTable(payload){
    const head = document.getElementById('trendHead');
    const body = document.getElementById('trendBody');
    const note = document.getElementById('trendTableNote');
    const dates = Array.isArray(payload.trend_dates) ? payload.trend_dates : [];
    const rows = Array.isArray(payload.trend_rows) ? payload.trend_rows : [];

    if (!dates.length || !rows.length){
      head.innerHTML = '';
      body.innerHTML = '<tr class="table-empty"><td colspan="2">No trend rows are available for this window.</td></tr>';
      note.textContent = '';
      return;
    }

    const visibleCount = Math.min(TREND_TABLE_DAYS, dates.length);
    const visibleDates = dates.slice(-visibleCount);
    note.textContent = `Showing the last ${visibleCount} date${visibleCount === 1 ? '' : 's'} in the selected window.`;

    head.innerHTML = `<tr>
      <th>Crisis Type</th>
      ${visibleDates.map((dateValue) => `<th class="numeric" title="${esc(dateValue)}">${esc(shortDateLabel(dateValue, true))}</th>`).join('')}
    </tr>`;

    body.innerHTML = rows.map((row) => {
      const series = Array.isArray(row.active_brands_series) ? row.active_brands_series.slice(-visibleCount) : [];
      return `<tr>
        <td><span class="tag-pill">${esc(row.display_tag || row.tag || '')}</span></td>
        ${series.map((value) => `<td class="numeric">${value || 0}</td>`).join('')}
      </tr>`;
    }).join('');
  }

  async function loadAvailableDates(){
    const payload = await fetchJson('/api/dates');
    availableDates = Array.isArray(payload.dates) ? payload.dates : [];
    const select = document.getElementById('dateSelect');
    const requestedDate = (params.get('date') || '').trim();
    const initialDate = availableDates.includes(requestedDate) ? requestedDate : (availableDates[0] || '');
    select.innerHTML = availableDates.map((dateValue) => (
      `<option value="${esc(dateValue)}">${esc(dateValue)}</option>`
    )).join('');
    if (initialDate) select.value = initialDate;
  }

  async function loadCrisisWindow(){
    const requestToken = ++activeRequestToken;
    const endDate = (document.getElementById('dateSelect').value || '').trim();
    const emptyPayload = {
      crisis_count: 0,
      affected_brand_count: 0,
      active_brand_count: 0,
      brand_day_count: 0,
      crises: [],
      trend_dates: [],
      trend_rows: [],
      selected_crisis: null,
    };

    if (!endDate){
      updateSummaryStats(emptyPayload);
      renderSummaryTable(emptyPayload);
      renderTrendChart(emptyPayload);
      renderTrendTable(emptyPayload);
      renderSelectedCrisis(emptyPayload);
      setStatus('No dates are available yet.');
      return;
    }

    const startDate = buildStartDate(endDate, currentDays);
    const query = new URLSearchParams({
      start_date: startDate,
      end_date: endDate,
    });
    if (selectedCrisisTag) query.set('crisis_tag', selectedCrisisTag);

    setStatus(`Loading crisis activity for ${startDate} to ${endDate}...`);
    try {
      const payload = await fetchJson(`/api/v1/insights/crisis_brand_impact?${query.toString()}`);
      if (requestToken !== activeRequestToken) return;
      if (!payload.selected_crisis && selectedCrisisTag){
        selectedCrisisTag = '';
      } else if (payload.selected_crisis && payload.selected_crisis.tag){
        selectedCrisisTag = payload.selected_crisis.tag;
      }
      updateSummaryStats(payload);
      renderSummaryTable(payload);
      renderTrendChart(payload);
      renderTrendTable(payload);
      renderSelectedCrisis(payload);
      setStatus(`Showing crisis activity from ${payload.window_start} to ${payload.window_end}.`);
    } catch (error) {
      if (requestToken !== activeRequestToken) return;
      updateSummaryStats(emptyPayload);
      renderSummaryTable(emptyPayload);
      renderTrendChart(emptyPayload);
      renderTrendTable(emptyPayload);
      renderSelectedCrisis(emptyPayload);
      setStatus(`Unable to load crisis activity: ${error.message}`, true);
    }
  }

  function bindControls(){
    const dateSelect = document.getElementById('dateSelect');
    dateSelect.addEventListener('change', () => {
      loadCrisisWindow();
    });

    const crisisSelect = document.getElementById('crisisSelect');
    crisisSelect.addEventListener('change', () => {
      selectedCrisisTag = String(crisisSelect.value || '').trim();
      loadCrisisWindow();
    });

    document.querySelectorAll('.lookback-btn').forEach((button) => {
      button.addEventListener('click', () => {
        const nextDays = Number(button.dataset.days || 0);
        if (!LOOKBACK_OPTIONS.includes(nextDays) || nextDays === currentDays) return;
        currentDays = nextDays;
        syncLookbackButtons();
        loadCrisisWindow();
      });
    });
  }

  async function init(){
    syncLookbackButtons();
    bindControls();
    try {
      await loadAvailableDates();
    } catch (error) {
      setStatus(`Unable to load available dates: ${error.message}`, true);
      return;
    }
    await loadCrisisWindow();
  }

  window.addEventListener('beforeunload', destroyTrendChart);
  init();
}

export async function prefetchParityTab({ getDirectUrl }) {
  return prefetchParityPage({ page, getDirectUrl });
}

export async function mountParityTab({ host, getDirectUrl, onHistoryChange }) {
  return mountParityPage({
    host,
    getDirectUrl,
    onHistoryChange,
    page,
    runInlineScript,
  });
}
