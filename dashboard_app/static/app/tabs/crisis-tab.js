import { ensureChartJs } from '../chart-runtime.js';
import { LOOKBACK_OPTIONS } from '../entity-config.js';
import { sharedFetch } from '../shared-fetch.js';
import { enableCardDragReorder } from '../card-reorder.js';

const TREND_COLORS = ['#ff3b30', '#ff6b57', '#f59e0b', '#facc15', '#d1d5db', '#bfdbfe', '#22c55e', '#a78bfa'];
const MAX_TREND_SERIES = 6;
const TREND_TABLE_DAYS = 7;
const prefetchRegistry = new Map();
const jsonCache = new Map();

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, (match) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  }[match]));
}

function formatInteger(value, fallback = '0') {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return fallback;
  return `${Math.round(numeric)}`;
}

function formatDays(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric <= 0) return '0d';
  const rounded = Math.round(numeric * 10) / 10;
  return `${Number.isInteger(rounded) ? rounded.toFixed(0) : rounded.toFixed(1)}d`;
}

function shortDateLabel(isoDate, includeYear = false) {
  const parts = String(isoDate || '').split('-');
  if (parts.length !== 3) return String(isoDate || '');
  const month = Number(parts[1]);
  const day = Number(parts[2]);
  if (includeYear) return `${month}/${day}/${parts[0].slice(2)}`;
  return `${month}/${day}`;
}

function buildStartDate(endIso, days) {
  const end = new Date(`${endIso}T00:00:00Z`);
  end.setUTCDate(end.getUTCDate() - (days - 1));
  return end.toISOString().slice(0, 10);
}

async function fetchJson(url) {
  if (jsonCache.has(url)) {
    const cached = jsonCache.get(url);
    return cached instanceof Promise ? await cached : cached;
  }

  const pending = (async () => {
    const response = await sharedFetch(url, { cache: 'default', credentials: 'same-origin' });
    if (!response.ok) {
      let detail = `HTTP ${response.status}`;
      try {
        const payload = await response.json();
        if (payload?.error) detail = payload.error;
      } catch (_error) {
        // Ignore non-JSON errors.
      }
      throw new Error(detail);
    }
    return response.json();
  })();

  jsonCache.set(url, pending);
  try {
    const data = await pending;
    jsonCache.set(url, data);
    return data;
  } catch (error) {
    jsonCache.delete(url);
    throw error;
  }
}

function renderActivePill(isActive) {
  return isActive
    ? '<span class="entity-pill" data-tone="high">Active</span>'
    : '<span class="entity-pill">Inactive</span>';
}

function createScaffold(config) {
  const lookbacks = LOOKBACK_OPTIONS.map((days) => `
    <button type="button" class="entity-lookback" data-lookback="${days}" aria-pressed="false">${days}d</button>
  `).join('');

  return `
    <div class="entity-tab crisis-tab">
      <section class="entity-hero">
        <article class="entity-hero-panel">
          <p class="entity-kicker">Native Crisis Dashboard Tab</p>
          <h2>${escapeHtml(config.title)}</h2>
          <p class="entity-copy">${escapeHtml(config.description)}</p>
          <div class="entity-summary-grid" data-role="summary-grid"></div>
        </article>
        <article class="entity-hero-panel entity-selected-spotlight" data-role="selected-spotlight" data-empty="true"></article>
      </section>

      <section class="entity-toolbar">
        <div class="entity-lookbacks">${lookbacks}</div>
        <label class="entity-field">
          <span>Window End Date</span>
          <select data-role="date-select"></select>
        </label>
        <label class="entity-field" style="min-width:260px;flex:1 1 260px;">
          <span>Crisis Tag</span>
          <select data-role="crisis-select">
            <option value="">Top crisis in window</option>
          </select>
        </label>
        <div class="entity-toolbar-actions">
          <button type="button" class="entity-action entity-action--reset" data-role="reset-button">Reset View</button>
          <a class="entity-action" data-role="open-link" target="_blank" rel="noopener">Open Full Dashboard</a>
        </div>
      </section>

      <div class="entity-feedback" data-role="feedback"></div>

      <article class="entity-card native-card--wide" data-card-id="crisis-activity-timeline">
        <header class="entity-card-header">
          <div>
            <h3>Crisis Activity Timeline</h3>
            <p data-role="trend-caption">Active affected brands by crisis tag through the selected window.</p>
          </div>
          <span class="entity-pill" data-role="trend-pill">Awaiting data</span>
        </header>
        <div class="entity-chart-wrap"><canvas data-role="trend-chart"></canvas></div>
        <div class="native-empty-note" data-role="trend-empty" hidden>No crisis trend rows are available for this window.</div>
      </article>

      <section class="entity-grid" data-card-group="crisis-grid">
        <article class="entity-card" data-card-id="crisis-summary">
          <header class="entity-card-header">
            <div>
              <h3>Crisis Summary</h3>
              <p>Click a row to focus the selected crisis.</p>
            </div>
            <span class="entity-pill" data-role="summary-pill">0 crises</span>
          </header>
          <div class="entity-table-scroll">
            <table class="entity-table crisis-mini-table">
              <thead>
                <tr>
                  <th>Crisis</th>
                  <th>Brands</th>
                  <th>Active</th>
                  <th>Avg Days</th>
                  <th>Longest</th>
                </tr>
              </thead>
              <tbody data-role="summary-body"></tbody>
            </table>
          </div>
        </article>

        <article class="entity-card" data-card-id="recent-trend-table">
          <header class="entity-card-header">
            <div>
              <h3>Recent Trend Table</h3>
              <p data-role="trend-note">Showing the last 7 dates in the selected window.</p>
            </div>
            <span class="entity-pill" data-role="trend-table-pill">0 rows</span>
          </header>
          <div class="entity-table-scroll">
            <table class="entity-table crisis-mini-table">
              <thead data-role="trend-head"></thead>
              <tbody data-role="trend-body"></tbody>
            </table>
          </div>
        </article>
      </section>

      <section class="entity-table-card">
        <div class="entity-table-head">
          <div>
            <h3>Affected Brands</h3>
            <p data-role="brand-caption">Selected crisis detail for the current window.</p>
          </div>
          <span class="entity-pill" data-role="brand-pill">0 brands</span>
        </div>
        <div class="entity-table-scroll">
          <table class="entity-table">
            <thead>
              <tr>
                <th>Brand</th>
                <th>Sector</th>
                <th>Active Days</th>
                <th>First Seen</th>
                <th>Last Seen</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody data-role="brand-body"></tbody>
          </table>
        </div>
      </section>
    </div>
  `;
}

class CrisisTabController {
  constructor({ host, tabId, shellConfig, getDirectUrl, onHistoryChange }) {
    this.host = host;
    this.tabId = tabId;
    this.shellConfig = shellConfig;
    this.getDirectUrl = getDirectUrl;
    this.onHistoryChange = onHistoryChange;
    this.config = {
      title: 'Crisis Radar',
      description: 'Native monitoring of crisis tags, trend lines, and the brands carrying them through the current window.',
      tabPath: Array.isArray(shellConfig.tabs) ? shellConfig.tabs.find((tab) => tab.id === tabId)?.path || '' : '',
    };
    this.state = {
      days: 30,
      date: '',
      crisisTag: '',
    };
    this.cleanups = [];
    this.chart = null;
    this.destroyed = false;
    this.availableDates = [];
    this.datesPromise = null;
    this.activeRequestToken = 0;
  }

  render() {
    this.host.innerHTML = createScaffold(this.config);
    this.root = this.host.firstElementChild;
    this.nodes = {
      lookbacks: Array.from(this.root.querySelectorAll('[data-lookback]')),
      dateSelect: this.root.querySelector('[data-role="date-select"]'),
      crisisSelect: this.root.querySelector('[data-role="crisis-select"]'),
      resetButton: this.root.querySelector('[data-role="reset-button"]'),
      openLink: this.root.querySelector('[data-role="open-link"]'),
      feedback: this.root.querySelector('[data-role="feedback"]'),
      summaryGrid: this.root.querySelector('[data-role="summary-grid"]'),
      selectedSpotlight: this.root.querySelector('[data-role="selected-spotlight"]'),
      trendCaption: this.root.querySelector('[data-role="trend-caption"]'),
      trendPill: this.root.querySelector('[data-role="trend-pill"]'),
      trendChart: this.root.querySelector('[data-role="trend-chart"]'),
      trendEmpty: this.root.querySelector('[data-role="trend-empty"]'),
      summaryBody: this.root.querySelector('[data-role="summary-body"]'),
      summaryPill: this.root.querySelector('[data-role="summary-pill"]'),
      trendHead: this.root.querySelector('[data-role="trend-head"]'),
      trendBody: this.root.querySelector('[data-role="trend-body"]'),
      trendNote: this.root.querySelector('[data-role="trend-note"]'),
      trendTablePill: this.root.querySelector('[data-role="trend-table-pill"]'),
      brandCaption: this.root.querySelector('[data-role="brand-caption"]'),
      brandPill: this.root.querySelector('[data-role="brand-pill"]'),
      brandBody: this.root.querySelector('[data-role="brand-body"]'),
    };
    this.cleanups.push(...enableCardDragReorder({
      root: this.root,
      storageNamespace: `${this.shellConfig.view || 'external'}:${this.tabId}`,
      onReorder: () => this.resizeCharts(),
    }));
  }

  on(target, type, listener, options) {
    target.addEventListener(type, listener, options);
    this.cleanups.push(() => target.removeEventListener(type, listener, options));
  }

  readUrlState() {
    const params = new URLSearchParams(this.getDirectUrl().search);
    const requestedDays = Number(params.get('days'));
    if (LOOKBACK_OPTIONS.includes(requestedDays)) {
      this.state.days = requestedDays;
    }
    this.state.date = String(params.get('date') || '').trim();
    this.state.crisisTag = String(params.get('crisis_tag') || '').trim();
  }

  updateUrl({ replace = true } = {}) {
    const nextUrl = new URL(this.getDirectUrl().href);
    if (this.state.days !== 30) nextUrl.searchParams.set('days', String(this.state.days));
    else nextUrl.searchParams.delete('days');
    if (this.state.date) nextUrl.searchParams.set('date', this.state.date);
    else nextUrl.searchParams.delete('date');
    if (this.state.crisisTag) nextUrl.searchParams.set('crisis_tag', this.state.crisisTag);
    else nextUrl.searchParams.delete('crisis_tag');
    this.onHistoryChange(nextUrl, { replace });
  }

  setFeedback(message, tone = '') {
    this.nodes.feedback.textContent = message || '';
    if (tone) this.nodes.feedback.dataset.state = tone;
    else delete this.nodes.feedback.dataset.state;
  }

  syncLookbacks() {
    this.nodes.lookbacks.forEach((button) => {
      button.setAttribute('aria-pressed', Number(button.dataset.lookback) === this.state.days ? 'true' : 'false');
    });
  }

  syncDateOptions() {
    this.nodes.dateSelect.innerHTML = this.availableDates.map((dateValue) => (
      `<option value="${escapeHtml(dateValue)}">${escapeHtml(dateValue)}</option>`
    )).join('');
    this.nodes.dateSelect.value = this.availableDates.includes(this.state.date)
      ? this.state.date
      : (this.availableDates[0] || '');
  }

  updateOpenLink() {
    const directUrl = this.getDirectUrl();
    this.nodes.openLink.href = `${directUrl.pathname}${directUrl.search}${directUrl.hash}`;
  }

  bind() {
    this.nodes.lookbacks.forEach((button) => {
      this.on(button, 'click', async () => {
        const nextDays = Number(button.dataset.lookback);
        if (!LOOKBACK_OPTIONS.includes(nextDays) || nextDays === this.state.days) return;
        this.state.days = nextDays;
        await this.refresh({ replaceUrl: false });
      });
    });

    this.on(this.nodes.dateSelect, 'change', async () => {
      this.state.date = this.nodes.dateSelect.value;
      await this.refresh({ replaceUrl: false });
    });

    this.on(this.nodes.crisisSelect, 'change', async () => {
      this.state.crisisTag = this.nodes.crisisSelect.value.trim();
      await this.refresh({ replaceUrl: false });
    });

    this.on(this.nodes.summaryBody, 'click', async (event) => {
      const row = event.target.closest('[data-crisis-tag]');
      if (!row) return;
      this.state.crisisTag = row.getAttribute('data-crisis-tag') || '';
      await this.refresh({ replaceUrl: false });
    });

    this.on(this.nodes.resetButton, 'click', async () => {
      this.state.days = 30;
      this.state.crisisTag = '';
      this.state.date = this.availableDates[0] || '';
      await this.refresh({ replaceUrl: false });
    });
  }

  async ensureDates() {
    if (this.datesPromise) {
      this.availableDates = await this.datesPromise;
      return this.availableDates;
    }
    this.datesPromise = (async () => {
      const payload = await fetchJson('/api/dates');
      return Array.isArray(payload.dates) ? payload.dates : [];
    })();
    this.availableDates = await this.datesPromise;
    return this.availableDates;
  }

  applySummary(payload) {
    const crisisRows = Array.isArray(payload.crises) ? payload.crises : [];
    const avgCrisisLength = crisisRows.length
      ? crisisRows.reduce((sum, row) => sum + Number(row.avg_active_days || 0), 0) / crisisRows.length
      : 0;

    const cards = [
      { label: 'Crisis Tags', value: formatInteger(payload.crisis_count) },
      { label: 'Affected Brands', value: formatInteger(payload.affected_brand_count) },
      { label: 'Active Brands', value: formatInteger(payload.active_brand_count) },
      { label: 'Average Duration', value: formatDays(avgCrisisLength) },
      { label: 'Brand Days', value: formatInteger(payload.brand_day_count) },
      { label: 'Window', value: payload.window_start && payload.window_end ? `${shortDateLabel(payload.window_start, true)} - ${shortDateLabel(payload.window_end, true)}` : 'N/A' },
    ];

    this.nodes.summaryGrid.innerHTML = cards.map((card) => `
      <div class="entity-stat">
        <p class="entity-stat-label">${escapeHtml(card.label)}</p>
        <p class="entity-stat-value">${escapeHtml(card.value)}</p>
      </div>
    `).join('');
  }

  applySpotlight(payload) {
    const selected = payload.selected_crisis;
    if (!selected) {
      this.nodes.selectedSpotlight.dataset.empty = 'true';
      this.nodes.selectedSpotlight.innerHTML = `
        <div>
          <p class="entity-kicker">Selected Crisis</p>
          <h3 style="margin:0 0 10px;">Choose a crisis from the summary table</h3>
          <p class="entity-copy" style="margin:0;">The spotlight panel fills in with affected-brand counts, active duration, and brand-level detail when you select a crisis.</p>
        </div>
      `;
      return;
    }

    this.nodes.selectedSpotlight.dataset.empty = 'false';
    this.nodes.selectedSpotlight.innerHTML = `
      <div class="entity-selected-header">
        <div>
          <h3>${escapeHtml(selected.display_tag || selected.tag || 'Selected crisis')}</h3>
          <p class="entity-selected-meta">Visible on ${formatInteger(selected.crisis_days)} dates in this window, backed by ${formatInteger(selected.total_negative_items)} tagged negative items.</p>
        </div>
        <span class="entity-pill" data-tone="high">${formatInteger(selected.brands_affected)} brands</span>
      </div>
      <div class="entity-selected-stats">
        <div class="entity-stat">
          <p class="entity-stat-label">Active Brands</p>
          <p class="entity-stat-value">${formatInteger(selected.active_brands_latest)}</p>
        </div>
        <div class="entity-stat">
          <p class="entity-stat-label">Average Active Days</p>
          <p class="entity-stat-value">${formatDays(selected.avg_active_days)}</p>
        </div>
        <div class="entity-stat">
          <p class="entity-stat-label">Longest Run</p>
          <p class="entity-stat-value">${formatDays(selected.longest_active_days)}</p>
        </div>
        <div class="entity-stat">
          <p class="entity-stat-label">Brand Days</p>
          <p class="entity-stat-value">${formatInteger(selected.brand_days)}</p>
        </div>
      </div>
    `;
  }

  renderSummaryTable(payload) {
    const rows = Array.isArray(payload.crises) ? payload.crises : [];
    const selectedKey = String(payload.selected_crisis?.tag || this.state.crisisTag || '').toLowerCase();
    this.nodes.summaryPill.textContent = `${rows.length} ${rows.length === 1 ? 'crisis' : 'crises'}`;

    if (!rows.length) {
      this.nodes.summaryBody.innerHTML = '<tr><td colspan="5" class="entity-table-empty">No crisis tags were active in this window.</td></tr>';
      return;
    }

    this.nodes.summaryBody.innerHTML = rows.map((row) => {
      const rowKey = String(row.tag || '').toLowerCase();
      return `
        <tr data-crisis-tag="${escapeHtml(row.tag || '')}" data-selected="${rowKey === selectedKey ? 'true' : 'false'}">
          <td><span class="entity-name-primary">${escapeHtml(row.display_tag || row.tag || '')}</span></td>
          <td>${formatInteger(row.brands_affected)}</td>
          <td>${formatInteger(row.active_brands_latest)}</td>
          <td>${formatDays(row.avg_active_days)}</td>
          <td>${formatDays(row.longest_active_days)}</td>
        </tr>
      `;
    }).join('');

  }

  syncCrisisSelect(payload) {
    const rows = Array.isArray(payload.crises) ? payload.crises : [];
    const options = ['<option value="">Top crisis in window</option>'];
    rows.forEach((row) => {
      options.push(`<option value="${escapeHtml(row.tag || '')}">${escapeHtml(row.display_tag || row.tag || '')}</option>`);
    });
    this.nodes.crisisSelect.innerHTML = options.join('');
    this.nodes.crisisSelect.value = payload.selected_crisis?.tag || this.state.crisisTag || '';
  }

  selectTrendRows(payload) {
    const rows = Array.isArray(payload.trend_rows) ? payload.trend_rows : [];
    const selectedKey = String(payload.selected_crisis?.tag || this.state.crisisTag || '').toLowerCase();
    const selectedRow = rows.find((row) => String(row.tag || '').toLowerCase() === selectedKey);
    let visibleRows = rows.slice(0, MAX_TREND_SERIES);
    if (selectedRow && !visibleRows.some((row) => String(row.tag || '').toLowerCase() === selectedKey)) {
      visibleRows = visibleRows.slice(0, Math.max(0, MAX_TREND_SERIES - 1));
      visibleRows.push(selectedRow);
    }
    return visibleRows;
  }

  renderTrendChart(payload) {
    const dates = Array.isArray(payload.trend_dates) ? payload.trend_dates : [];
    const rows = this.selectTrendRows(payload);
    if (this.chart) {
      this.chart.destroy();
      this.chart = null;
    }

    if (!dates.length || !rows.length) {
      this.nodes.trendChart.hidden = true;
      this.nodes.trendEmpty.hidden = false;
      this.nodes.trendPill.textContent = 'No trend rows';
      return;
    }

    this.nodes.trendChart.hidden = false;
    this.nodes.trendEmpty.hidden = true;
    this.nodes.trendPill.textContent = `${rows.length} active series`;
    this.nodes.trendCaption.textContent = payload.window_start && payload.window_end
      ? `Showing activity from ${payload.window_start} to ${payload.window_end}.`
      : 'Active affected brands by crisis tag through the selected window.';

    this.chart = new window.Chart(this.nodes.trendChart.getContext('2d'), {
      type: 'line',
      data: {
        labels: dates.map((dateValue) => shortDateLabel(dateValue)),
        datasets: rows.map((row, index) => ({
          label: row.display_tag || row.tag || `Crisis ${index + 1}`,
          data: Array.isArray(row.active_brands_series) ? row.active_brands_series : [],
          borderColor: TREND_COLORS[index % TREND_COLORS.length],
          backgroundColor: TREND_COLORS[index % TREND_COLORS.length],
          pointRadius: 0,
          pointHoverRadius: 4,
          borderWidth: 3,
          tension: 0.24,
        })),
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
            },
          },
          tooltip: {
            callbacks: {
              title: (items) => {
                const idx = items?.[0]?.dataIndex ?? 0;
                return dates[idx] || '';
              },
              label: (context) => `${context.dataset?.label || 'Series'}: ${context.parsed?.y ?? 0}`,
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

  renderTrendTable(payload) {
    const dates = Array.isArray(payload.trend_dates) ? payload.trend_dates : [];
    const rows = Array.isArray(payload.trend_rows) ? payload.trend_rows : [];
    if (!dates.length || !rows.length) {
      this.nodes.trendHead.innerHTML = '';
      this.nodes.trendBody.innerHTML = '<tr><td colspan="2" class="entity-table-empty">No trend rows are available for this window.</td></tr>';
      this.nodes.trendNote.textContent = '';
      this.nodes.trendTablePill.textContent = '0 rows';
      return;
    }

    const visibleCount = Math.min(TREND_TABLE_DAYS, dates.length);
    const visibleDates = dates.slice(-visibleCount);
    this.nodes.trendNote.textContent = `Showing the last ${visibleCount} date${visibleCount === 1 ? '' : 's'} in the selected window.`;
    this.nodes.trendTablePill.textContent = `${rows.length} ${rows.length === 1 ? 'row' : 'rows'}`;
    this.nodes.trendHead.innerHTML = `<tr>
      <th>Crisis</th>
      ${visibleDates.map((dateValue) => `<th>${escapeHtml(shortDateLabel(dateValue, true))}</th>`).join('')}
    </tr>`;
    this.nodes.trendBody.innerHTML = rows.map((row) => {
      const series = Array.isArray(row.active_brands_series) ? row.active_brands_series.slice(-visibleCount) : [];
      return `<tr>
        <td><span class="entity-name-primary">${escapeHtml(row.display_tag || row.tag || '')}</span></td>
        ${series.map((value) => `<td>${formatInteger(value)}</td>`).join('')}
      </tr>`;
    }).join('');
  }

  renderBrandTable(payload) {
    const selected = payload.selected_crisis;
    const brands = Array.isArray(selected?.brands) ? selected.brands : [];
    this.nodes.brandPill.textContent = `${brands.length} ${brands.length === 1 ? 'brand' : 'brands'}`;
    this.nodes.brandCaption.textContent = selected
      ? `Brand detail for ${selected.display_tag || selected.tag || 'the selected crisis'}.`
      : 'Select a crisis above to inspect the affected brands.';

    if (!brands.length) {
      this.nodes.brandBody.innerHTML = '<tr><td colspan="6" class="entity-table-empty">No brands were attached to this crisis in the selected window.</td></tr>';
      return;
    }

    this.nodes.brandBody.innerHTML = brands.map((brand) => `
      <tr>
        <td>
          <div class="entity-name-cell">
            <span class="entity-name-primary">${escapeHtml(brand.brand || '')}</span>
          </div>
        </td>
        <td>${escapeHtml(brand.sector || 'Unassigned')}</td>
        <td>${formatInteger(brand.days_affected)}</td>
        <td>${escapeHtml(brand.first_seen_date || '')}</td>
        <td>${escapeHtml(brand.last_seen_date || '')}</td>
        <td>${renderActivePill(!!brand.active_on_window_end)}</td>
      </tr>
    `).join('');
  }

  applyPayload(payload) {
    if (!payload.selected_crisis && this.state.crisisTag) {
      this.state.crisisTag = '';
    } else if (payload.selected_crisis?.tag) {
      this.state.crisisTag = payload.selected_crisis.tag;
    }

    this.applySummary(payload);
    this.applySpotlight(payload);
    this.syncCrisisSelect(payload);
    this.renderSummaryTable(payload);
    this.renderTrendChart(payload);
    this.renderTrendTable(payload);
    this.renderBrandTable(payload);
    this.setFeedback(payload.window_start && payload.window_end
      ? `Showing crisis activity from ${payload.window_start} to ${payload.window_end}.`
      : 'Showing crisis activity for the current window.');
  }

  async refresh({ replaceUrl = true } = {}) {
    this.syncLookbacks();
    if (!this.availableDates.length) {
      this.applyPayload({
        crisis_count: 0,
        affected_brand_count: 0,
        active_brand_count: 0,
        brand_day_count: 0,
        crises: [],
        trend_dates: [],
        trend_rows: [],
        selected_crisis: null,
      });
      this.setFeedback('No dates are available yet.', 'error');
      return;
    }

    if (!this.availableDates.includes(this.state.date)) {
      this.state.date = this.availableDates[0] || '';
    }
    this.syncDateOptions();
    this.updateOpenLink();

    if (!this.state.date) {
      this.setFeedback('No dates are available yet.', 'error');
      return;
    }

    const requestToken = ++this.activeRequestToken;
    const startDate = buildStartDate(this.state.date, this.state.days);
    const query = new URLSearchParams({
      start_date: startDate,
      end_date: this.state.date,
    });
    if (this.state.crisisTag) query.set('crisis_tag', this.state.crisisTag);

    this.setFeedback(`Loading crisis activity for ${startDate} to ${this.state.date}...`);
    try {
      const payload = await fetchJson(`/api/v1/insights/crisis_brand_impact?${query.toString()}`);
      if (requestToken !== this.activeRequestToken || this.destroyed) return;
      this.applyPayload(payload);
      this.updateUrl({ replace: replaceUrl });
      this.updateOpenLink();
    } catch (error) {
      if (requestToken !== this.activeRequestToken || this.destroyed) return;
      this.applyPayload({
        crisis_count: 0,
        affected_brand_count: 0,
        active_brand_count: 0,
        brand_day_count: 0,
        crises: [],
        trend_dates: [],
        trend_rows: [],
        selected_crisis: null,
      });
      this.setFeedback(`Unable to load crisis activity: ${error.message}`, 'error');
    }
  }

  async init() {
    this.readUrlState();
    this.render();
    this.bind();
    await ensureChartJs();
    this.availableDates = await this.ensureDates();
    if (!this.availableDates.includes(this.state.date)) {
      this.state.date = this.availableDates[0] || '';
    }
    this.syncDateOptions();
    await this.refresh({ replaceUrl: false });
  }

  resizeCharts() {
    if (!this.chart) return;
    try {
      this.chart.resize();
    } catch (_error) {
      // Ignore resize issues while hidden.
    }
  }

  show() {
    this.host.hidden = false;
    window.requestAnimationFrame(() => this.resizeCharts());
  }

  hide() {
    this.host.hidden = true;
  }

  destroy() {
    this.destroyed = true;
    this.cleanups.splice(0).forEach((cleanup) => {
      try {
        cleanup();
      } catch (_error) {
        // Ignore teardown failures.
      }
    });
    if (this.chart) {
      try {
        this.chart.destroy();
      } catch (_error) {
        // Ignore Chart.js teardown failures.
      }
    }
    this.chart = null;
    this.host.innerHTML = '';
    this.host.hidden = true;
  }
}

export async function prefetchCrisisTab({ tabId, shellConfig, getDirectUrl }) {
  const cacheKey = `${shellConfig.view || 'external'}:${tabId}`;
  if (prefetchRegistry.has(cacheKey)) return prefetchRegistry.get(cacheKey);

  const task = (async () => {
    await ensureChartJs();
    const payload = await fetchJson('/api/dates');
    const dates = Array.isArray(payload.dates) ? payload.dates : [];
    const directUrl = getDirectUrl();
    const requestedDate = String(directUrl.searchParams.get('date') || '').trim();
    const endDate = dates.includes(requestedDate) ? requestedDate : (dates[0] || '');
    if (!endDate) return;
    const requestedDays = Number(directUrl.searchParams.get('days'));
    const days = LOOKBACK_OPTIONS.includes(requestedDays) ? requestedDays : 30;
    const crisisTag = String(directUrl.searchParams.get('crisis_tag') || '').trim();
    const query = new URLSearchParams({
      start_date: buildStartDate(endDate, days),
      end_date: endDate,
    });
    if (crisisTag) query.set('crisis_tag', crisisTag);
    await fetchJson(`/api/v1/insights/crisis_brand_impact?${query.toString()}`);
  })();

  prefetchRegistry.set(cacheKey, task);
  try {
    await task;
  } catch (error) {
    prefetchRegistry.delete(cacheKey);
    throw error;
  }
}

export async function mountCrisisTab({ host, getDirectUrl, onHistoryChange, tabId, shellConfig }) {
  const controller = new CrisisTabController({
    host,
    getDirectUrl,
    onHistoryChange,
    tabId,
    shellConfig,
  });
  await controller.init();
  return {
    show: () => controller.show(),
    hide: () => controller.hide(),
    destroy: () => controller.destroy(),
  };
}
