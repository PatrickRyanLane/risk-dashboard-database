import { ensureChartJs } from '../chart-runtime.js';
import { LOOKBACK_OPTIONS } from '../entity-config.js';
import { runInternalRefresh } from '../internal-refresh.js';
import { clearAllNativeDataStores } from '../native-stores.js';
import { FEATURE_LABELS, FEATURE_ORDER_SENTIMENT, getSectorStore } from '../sector-store.js';
import { enableCardDragReorder } from '../card-reorder.js';

const FEATURE_COLORS = ['#58dbed', '#ff5e57', '#ff7a59', '#ff9f43', '#ff6f91', '#e63946'];
const CONTROL_COLORS = ['#5ad0e1', '#3cbacb', '#2d9cdb', '#247f9a', '#1a6278', '#134c5e'];
const prefetchRegistry = new Map();

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, (match) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  }[match]));
}

function formatPercent(value, digits = 1, fallback = 'N/A') {
  if (value == null || Number.isNaN(value)) return fallback;
  return `${(value * 100).toFixed(digits)}%`;
}

function formatSignedPercent(value, fallback = 'Pending') {
  if (value == null || Number.isNaN(value)) return fallback;
  const prefix = value > 0 ? '+' : '';
  return `${prefix}${value.toFixed(2)}%`;
}

function formatInteger(value, fallback = '0') {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return fallback;
  return `${Math.round(numeric)}`;
}

function riskTone(label) {
  if (label === 'High') return 'high';
  if (label === 'Medium') return 'medium';
  if (label === 'Low') return 'low';
  return 'neutral';
}

function scheduleIdle(callback) {
  if (window.requestIdleCallback) {
    return window.requestIdleCallback(callback);
  }
  return window.setTimeout(callback, 0);
}

function shortDateLabel(isoDate) {
  const parts = String(isoDate || '').split('-');
  if (parts.length !== 3) return String(isoDate || '');
  return `${Number(parts[1])}/${Number(parts[2])}`;
}

function createScaffold(config) {
  const lookbacks = LOOKBACK_OPTIONS.map((days) => `
    <button type="button" class="entity-lookback" data-lookback="${days}" aria-pressed="false">${days}d</button>
  `).join('');

  return `
    <div class="entity-tab sector-tab">
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
          <span>Active Date</span>
          <select data-role="date-select"></select>
        </label>
        <label class="entity-field" style="min-width:220px;flex:1 1 220px;">
          <span>Search Sectors</span>
          <input type="search" data-role="query-input" placeholder="Search by sector name" />
        </label>
        <label class="entity-field">
          <span>Sector Filter</span>
          <select data-role="sector-select">
            <option value="">All sectors</option>
          </select>
        </label>
        <label class="entity-field">
          <span>Company Size</span>
          <select data-role="size-select">
            <option value="all">All companies</option>
            <option value="fortune500">Fortune 500</option>
            <option value="fortune1000">Fortune 1000</option>
            <option value="forbes">Forbes</option>
          </select>
        </label>
        <div class="entity-toolbar-actions">
          <button type="button" class="entity-action" data-role="crisis-toggle">Crisis Brands Only</button>
          ${config.isInternal ? '<button type="button" class="entity-action" data-role="refresh-button">Refresh Data</button>' : ''}
          <button type="button" class="entity-action entity-action--reset" data-role="reset-button">Reset View</button>
          <a class="entity-action" data-role="open-link" target="_blank" rel="noopener">Open Full Dashboard</a>
        </div>
      </section>

      <div class="entity-feedback" data-role="feedback"></div>

      <article class="entity-card native-card--wide" data-card-id="all-surface-sector-pressure">
        <header class="entity-card-header">
          <div>
            <h3>All-Surface Sector Pressure</h3>
            <p data-role="timeline-caption">Combined news and search pressure through the selected lookback window.</p>
          </div>
          <span class="entity-pill" data-role="timeline-pill">Awaiting data</span>
        </header>
        <div class="entity-chart-wrap"><canvas data-role="timeline-chart"></canvas></div>
      </article>

      <section class="entity-grid">
        <div class="entity-grid-primary" data-card-group="sector-primary">
          <article class="entity-card" data-card-id="news-mix">
            <header class="entity-card-header">
              <div>
                <h3>News Mix</h3>
                <p data-role="news-caption">Positive, neutral, and negative share across the selected scope.</p>
              </div>
              <span class="entity-pill" data-role="news-pill">Awaiting data</span>
            </header>
            <div class="entity-chart-wrap"><canvas data-role="news-chart"></canvas></div>
          </article>

          <article class="entity-card" data-card-id="feature-negative-share">
            <header class="entity-card-header">
              <div>
                <h3>Feature Negative Share</h3>
                <p data-role="feature-caption">Feature-level share of total visible search surface.</p>
              </div>
              <span class="entity-pill" data-role="feature-pill">Awaiting data</span>
            </header>
            <div class="entity-chart-wrap"><canvas data-role="feature-chart"></canvas></div>
          </article>
        </div>

        <div class="entity-grid-secondary" data-card-group="sector-secondary">
          <article class="entity-card" data-card-id="feature-control-share">
            <header class="entity-card-header">
              <div>
                <h3>Feature Control Share</h3>
                <p data-role="control-caption">Controlled share across organic and feature surfaces.</p>
              </div>
              <span class="entity-pill" data-role="control-pill">Awaiting data</span>
            </header>
            <div class="entity-chart-wrap"><canvas data-role="control-chart"></canvas></div>
          </article>

          <article class="entity-card" data-card-id="sector-signal-leaderboard">
            <header class="entity-card-header">
              <div>
                <h3>Signal Leaderboard</h3>
                <p data-role="leaderboard-caption">Highest combined sector pressure on the active date.</p>
              </div>
              <span class="entity-pill" data-role="leaderboard-pill">Awaiting data</span>
            </header>
            <div class="entity-chart-wrap"><canvas data-role="leaderboard-chart"></canvas></div>
          </article>
        </div>
      </section>

      <section class="entity-table-card">
        <div class="entity-table-head">
          <div>
            <h3>Sector Watchlist</h3>
            <p data-role="table-caption">Click a sector row to focus the charts and spotlight.</p>
          </div>
          <span class="entity-pill" data-role="table-pill">0 sectors</span>
        </div>
        <div class="entity-table-scroll">
          <table class="entity-table">
            <thead data-role="table-head"></thead>
            <tbody data-role="table-body"></tbody>
          </table>
        </div>
      </section>
    </div>
  `;
}

class SectorTabController {
  constructor({ host, tabId, shellConfig, getDirectUrl, onHistoryChange }) {
    this.host = host;
    this.tabId = tabId;
    this.shellConfig = shellConfig;
    this.getDirectUrl = getDirectUrl;
    this.onHistoryChange = onHistoryChange;
    this.config = {
      view: shellConfig.view || 'external',
      tabId,
      isInternal: shellConfig.view === 'internal',
      title: 'Sector Pressure Map',
      description: 'Native sector-level crisis monitoring with shared data caches, instant tab switching, and one shell-owned state model.',
      tabPath: Array.isArray(shellConfig.tabs) ? shellConfig.tabs.find((tab) => tab.id === tabId)?.path || '' : '',
    };
    this.store = getSectorStore(this.config);
    this.cleanups = [];
    this.charts = {};
    this.destroyed = false;
    this.stockScheduled = false;
    this.snapshot = null;
    this.state = {
      days: 30,
      date: '',
      query: '',
      sectorFilter: '',
      companySizeFilter: 'all',
      crisisOnly: false,
      selectedSector: '',
      sortKey: 'riskScore',
      sortDir: 'desc',
    };
  }

  render() {
    this.host.innerHTML = createScaffold(this.config);
    this.root = this.host.firstElementChild;
    this.nodes = {
      lookbacks: Array.from(this.root.querySelectorAll('[data-lookback]')),
      dateSelect: this.root.querySelector('[data-role="date-select"]'),
      queryInput: this.root.querySelector('[data-role="query-input"]'),
      sectorSelect: this.root.querySelector('[data-role="sector-select"]'),
      sizeSelect: this.root.querySelector('[data-role="size-select"]'),
      crisisToggle: this.root.querySelector('[data-role="crisis-toggle"]'),
      refreshButton: this.root.querySelector('[data-role="refresh-button"]'),
      resetButton: this.root.querySelector('[data-role="reset-button"]'),
      openLink: this.root.querySelector('[data-role="open-link"]'),
      feedback: this.root.querySelector('[data-role="feedback"]'),
      summaryGrid: this.root.querySelector('[data-role="summary-grid"]'),
      selectedSpotlight: this.root.querySelector('[data-role="selected-spotlight"]'),
      timelineCaption: this.root.querySelector('[data-role="timeline-caption"]'),
      timelinePill: this.root.querySelector('[data-role="timeline-pill"]'),
      newsCaption: this.root.querySelector('[data-role="news-caption"]'),
      newsPill: this.root.querySelector('[data-role="news-pill"]'),
      featureCaption: this.root.querySelector('[data-role="feature-caption"]'),
      featurePill: this.root.querySelector('[data-role="feature-pill"]'),
      controlCaption: this.root.querySelector('[data-role="control-caption"]'),
      controlPill: this.root.querySelector('[data-role="control-pill"]'),
      leaderboardCaption: this.root.querySelector('[data-role="leaderboard-caption"]'),
      leaderboardPill: this.root.querySelector('[data-role="leaderboard-pill"]'),
      tableCaption: this.root.querySelector('[data-role="table-caption"]'),
      tablePill: this.root.querySelector('[data-role="table-pill"]'),
      tableHead: this.root.querySelector('[data-role="table-head"]'),
      tableBody: this.root.querySelector('[data-role="table-body"]'),
      timelineChart: this.root.querySelector('[data-role="timeline-chart"]'),
      newsChart: this.root.querySelector('[data-role="news-chart"]'),
      featureChart: this.root.querySelector('[data-role="feature-chart"]'),
      controlChart: this.root.querySelector('[data-role="control-chart"]'),
      leaderboardChart: this.root.querySelector('[data-role="leaderboard-chart"]'),
    };
    this.cleanups.push(...enableCardDragReorder({
      root: this.root,
      storageNamespace: `${this.config.view}:${this.config.tabId}`,
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
    if (LOOKBACK_OPTIONS.includes(requestedDays)) this.state.days = requestedDays;
    this.state.date = String(params.get('date') || '').trim();
    this.state.query = String(params.get('q') || '').trim();
    this.state.sectorFilter = String(params.get('sector') || '').trim();
    this.state.companySizeFilter = String(params.get('size') || 'all').trim() || 'all';
    this.state.crisisOnly = params.get('crisis_only') === '1';
    this.state.selectedSector = String(params.get('selected') || '').trim();
  }

  updateUrl({ replace = true } = {}) {
    const nextUrl = new URL(this.getDirectUrl().href);
    if (this.state.days !== 30) nextUrl.searchParams.set('days', String(this.state.days));
    else nextUrl.searchParams.delete('days');
    if (this.state.date) nextUrl.searchParams.set('date', this.state.date);
    else nextUrl.searchParams.delete('date');
    if (this.state.query) nextUrl.searchParams.set('q', this.state.query);
    else nextUrl.searchParams.delete('q');
    if (this.state.sectorFilter) nextUrl.searchParams.set('sector', this.state.sectorFilter);
    else nextUrl.searchParams.delete('sector');
    if (this.state.companySizeFilter && this.state.companySizeFilter !== 'all') {
      nextUrl.searchParams.set('size', this.state.companySizeFilter);
    } else {
      nextUrl.searchParams.delete('size');
    }
    if (this.state.crisisOnly) nextUrl.searchParams.set('crisis_only', '1');
    else nextUrl.searchParams.delete('crisis_only');
    if (this.state.selectedSector) nextUrl.searchParams.set('selected', this.state.selectedSector);
    else nextUrl.searchParams.delete('selected');
    this.onHistoryChange(nextUrl, { replace });
  }

  setFeedback(message, tone = '') {
    this.nodes.feedback.textContent = message || '';
    if (tone) this.nodes.feedback.dataset.state = tone;
    else delete this.nodes.feedback.dataset.state;
  }

  updateOpenLink() {
    const directUrl = this.getDirectUrl();
    this.nodes.openLink.href = `${directUrl.pathname}${directUrl.search}${directUrl.hash}`;
  }

  syncControls(snapshot) {
    this.nodes.lookbacks.forEach((button) => {
      button.setAttribute('aria-pressed', Number(button.dataset.lookback) === this.state.days ? 'true' : 'false');
    });

    this.nodes.queryInput.value = this.state.query;
    this.nodes.sizeSelect.value = this.state.companySizeFilter;
    this.nodes.crisisToggle.setAttribute('aria-pressed', this.state.crisisOnly ? 'true' : 'false');
    this.nodes.crisisToggle.dataset.variant = this.state.crisisOnly ? 'active' : 'ghost';

    this.nodes.dateSelect.innerHTML = snapshot.dateOptions.map((dateValue) => (
      `<option value="${escapeHtml(dateValue)}">${escapeHtml(dateValue)}</option>`
    )).join('');
    this.nodes.dateSelect.value = snapshot.activeDate;

    const sectorOptions = ['<option value="">All sectors</option>'];
    snapshot.sectorOptions.forEach((sectorName) => {
      sectorOptions.push(`<option value="${escapeHtml(sectorName)}">${escapeHtml(sectorName)}</option>`);
    });
    this.nodes.sectorSelect.innerHTML = sectorOptions.join('');
    this.nodes.sectorSelect.value = this.state.sectorFilter;
    this.updateOpenLink();
  }

  bind() {
    this.nodes.lookbacks.forEach((button) => {
      this.on(button, 'click', async () => {
        const nextDays = Number(button.dataset.lookback);
        if (!LOOKBACK_OPTIONS.includes(nextDays) || nextDays === this.state.days) return;
        this.state.days = nextDays;
        await this.load({ replaceUrl: false });
      });
    });

    this.on(this.nodes.dateSelect, 'change', async () => {
      this.state.date = this.nodes.dateSelect.value;
      await this.refreshView({ replaceUrl: false });
    });

    this.on(this.nodes.queryInput, 'input', async () => {
      this.state.query = this.nodes.queryInput.value.trim();
      await this.refreshView({ replaceUrl: false, updateCharts: true });
    });

    this.on(this.nodes.sectorSelect, 'change', async () => {
      this.state.sectorFilter = this.nodes.sectorSelect.value.trim();
      if (this.state.selectedSector && this.state.selectedSector !== this.state.sectorFilter) {
        this.state.selectedSector = '';
      }
      await this.refreshView({ replaceUrl: false, updateCharts: true });
    });

    this.on(this.nodes.sizeSelect, 'change', async () => {
      this.state.companySizeFilter = this.nodes.sizeSelect.value || 'all';
      await this.refreshView({ replaceUrl: false, updateCharts: true });
    });

    this.on(this.nodes.crisisToggle, 'click', async () => {
      this.state.crisisOnly = !this.state.crisisOnly;
      await this.refreshView({ replaceUrl: false, updateCharts: true });
    });

    this.on(this.nodes.resetButton, 'click', async () => {
      this.state.query = '';
      this.state.sectorFilter = '';
      this.state.companySizeFilter = 'all';
      this.state.crisisOnly = false;
      this.state.selectedSector = '';
      this.state.sortKey = 'riskScore';
      this.state.sortDir = 'desc';
      this.state.date = this.snapshot?.dateOptions[this.snapshot.dateOptions.length - 1] || '';
      await this.refreshView({ replaceUrl: false, updateCharts: true });
    });

    if (this.nodes.refreshButton) {
      this.on(this.nodes.refreshButton, 'click', async () => {
        try {
          this.setFeedback('Refreshing internal aggregates...');
          const status = await runInternalRefresh(this.nodes.feedback);
          clearAllNativeDataStores();
          this.store = getSectorStore(this.config);
          this.stockScheduled = false;
          await this.load({ replaceUrl: false });
          this.setFeedback(status === 'ok' ? 'Refresh complete.' : `Refresh status: ${status}.`);
        } catch (error) {
          this.setFeedback(error?.message || 'Refresh failed.', 'error');
        }
      });
    }

    this.on(this.nodes.tableHead, 'click', async (event) => {
      const cell = event.target.closest('[data-sort-key]');
      if (!cell) return;
      const sortKey = cell.getAttribute('data-sort-key');
      if (!sortKey) return;
      if (this.state.sortKey === sortKey) {
        this.state.sortDir = this.state.sortDir === 'asc' ? 'desc' : 'asc';
      } else {
        this.state.sortKey = sortKey;
        this.state.sortDir = sortKey === 'sector' ? 'asc' : 'desc';
      }
      await this.refreshView({ replaceUrl: false, updateCharts: false });
    });

    this.on(this.nodes.tableBody, 'click', async (event) => {
      const row = event.target.closest('[data-sector]');
      if (!row) return;
      const sectorName = row.getAttribute('data-sector') || '';
      this.state.selectedSector = this.state.selectedSector === sectorName ? '' : sectorName;
      await this.refreshView({ replaceUrl: false, updateCharts: true });
    });
  }

  renderSummary(snapshot) {
    const cards = [
      { label: 'Visible Sectors', value: formatInteger(snapshot.summary.visibleSectorCount) },
      { label: 'Brands In Scope', value: formatInteger(snapshot.summary.visibleBrandCount) },
      { label: 'Average News Negative', value: formatPercent(snapshot.summary.avgNewsNeg) },
      { label: 'Average All-Surface Negative', value: formatPercent(snapshot.summary.avgSurfaceNeg) },
      { label: 'Average All-Surface Control', value: formatPercent(snapshot.summary.avgSurfaceCtrl) },
      { label: 'Active Date', value: snapshot.activeDate || 'N/A' },
    ];

    this.nodes.summaryGrid.innerHTML = cards.map((card) => `
      <div class="entity-stat">
        <p class="entity-stat-label">${escapeHtml(card.label)}</p>
        <p class="entity-stat-value">${escapeHtml(card.value)}</p>
      </div>
    `).join('');
  }

  renderSpotlight(snapshot) {
    const selected = snapshot.selectedRow;
    if (!selected) {
      this.nodes.selectedSpotlight.dataset.empty = 'true';
      this.nodes.selectedSpotlight.innerHTML = `
        <div>
          <p class="entity-kicker">Sector Spotlight</p>
          <h3 style="margin:0 0 10px;">Select a sector to lock the chart scope</h3>
          <p class="entity-copy" style="margin:0;">${escapeHtml(snapshot.summary.hint)} Click any table row to pin the spotlight and keep the charts focused while you switch tabs.</p>
        </div>
      `;
      return;
    }

    this.nodes.selectedSpotlight.dataset.empty = 'false';
    this.nodes.selectedSpotlight.innerHTML = `
      <div class="entity-selected-header">
        <div>
          <h3>${escapeHtml(selected.sector)}</h3>
          <p class="entity-selected-meta">${formatInteger(selected.brandCount)} visible brands, ${formatInteger(selected.crisisBrandCount)} crisis brands in scope on ${escapeHtml(snapshot.activeDate)}.</p>
        </div>
        <span class="entity-pill" data-tone="${riskTone(selected.riskTone)}">${escapeHtml(selected.riskTone)}</span>
      </div>
      <div class="entity-selected-stats">
        <div class="entity-stat">
          <p class="entity-stat-label">All-Surface Negative</p>
          <p class="entity-stat-value">${formatPercent(selected.negSerpAll)}</p>
        </div>
        <div class="entity-stat">
          <p class="entity-stat-label">All-Surface Control</p>
          <p class="entity-stat-value">${formatPercent(selected.ctrlPctAll)}</p>
        </div>
        <div class="entity-stat">
          <p class="entity-stat-label">News Negative</p>
          <p class="entity-stat-value">${formatPercent(selected.negNews)}</p>
        </div>
        <div class="entity-stat">
          <p class="entity-stat-label">7-Day Stock</p>
          <p class="entity-stat-value">${formatSignedPercent(selected.avgStockChange)}</p>
        </div>
      </div>
    `;
  }

  sortRows(rows) {
    const sorted = rows.slice();
    const direction = this.state.sortDir === 'asc' ? 1 : -1;
    sorted.sort((left, right) => {
      const leftValue = left[this.state.sortKey];
      const rightValue = right[this.state.sortKey];
      if (typeof leftValue === 'string' || typeof rightValue === 'string') {
        return direction * String(leftValue || '').localeCompare(String(rightValue || ''));
      }
      const safeLeft = Number.isFinite(leftValue) ? leftValue : -Infinity;
      const safeRight = Number.isFinite(rightValue) ? rightValue : -Infinity;
      return direction * (safeLeft - safeRight);
    });
    return sorted;
  }

  renderTable(snapshot) {
    const columns = [
      { key: 'sector', label: 'Sector' },
      { key: 'brandCount', label: 'Brands' },
      { key: 'avgDailyChange', label: 'Daily Stock' },
      { key: 'avgStockChange', label: '7D Stock' },
      { key: 'negNews', label: 'News Neg' },
      { key: 'negSerp', label: 'Organic Neg' },
      { key: 'ctrlPct', label: 'Organic Ctrl' },
      { key: 'negSerpAll', label: 'All-Surface Neg' },
      { key: 'ctrlPctAll', label: 'All-Surface Ctrl' },
      { key: 'riskScore', label: 'Signal' },
    ];
    this.nodes.tableHead.innerHTML = `<tr>${columns.map((column) => `
      <th data-sort-key="${column.key}">
        ${escapeHtml(column.label)}
        ${this.state.sortKey === column.key ? `<span class="native-sort-indicator">${this.state.sortDir === 'asc' ? '↑' : '↓'}</span>` : ''}
      </th>
    `).join('')}</tr>`;

    const rows = this.sortRows(snapshot.filteredRows);
    this.nodes.tablePill.textContent = `${rows.length} ${rows.length === 1 ? 'sector' : 'sectors'}`;
    this.nodes.tableCaption.textContent = snapshot.selectedSector
      ? `Charts are pinned to ${snapshot.selectedSector}.`
      : 'Click a sector row to focus the charts and spotlight.';

    if (!rows.length) {
      this.nodes.tableBody.innerHTML = '<tr><td colspan="10" class="entity-table-empty">No sectors matched the current filters.</td></tr>';
      return;
    }

    this.nodes.tableBody.innerHTML = rows.map((row) => `
      <tr data-sector="${escapeHtml(row.sector)}" data-selected="${row.sector === snapshot.selectedSector ? 'true' : 'false'}">
        <td>
          <div class="entity-name-cell">
            <span class="entity-name-primary">${escapeHtml(row.sector)}</span>
            <span class="entity-name-secondary">${formatInteger(row.brandCount)} brands</span>
          </div>
        </td>
        <td>${formatInteger(row.brandCount)}</td>
        <td class="entity-number" data-tone="${row.avgDailyChange == null ? 'neutral' : (row.avgDailyChange >= 0 ? 'positive' : 'negative')}">${formatSignedPercent(row.avgDailyChange)}</td>
        <td class="entity-number" data-tone="${row.avgStockChange == null ? 'neutral' : (row.avgStockChange >= 0 ? 'positive' : 'negative')}">${formatSignedPercent(row.avgStockChange)}</td>
        <td>${formatPercent(row.negNews)}</td>
        <td>${formatPercent(row.negSerp)}</td>
        <td>${formatPercent(row.ctrlPct)}</td>
        <td>${formatPercent(row.negSerpAll)}</td>
        <td>${formatPercent(row.ctrlPctAll)}</td>
        <td>
          <div class="entity-score-bar">
            <span class="entity-score-fill" style="width:${Math.max(0, Math.min(100, Math.round(row.riskScore * 100)))}%"></span>
          </div>
        </td>
      </tr>
    `).join('');

  }

  destroyCharts() {
    Object.values(this.charts).forEach((chart) => {
      try {
        chart.destroy();
      } catch (_error) {
        // Ignore Chart.js teardown failures.
      }
    });
    this.charts = {};
  }

  renderCharts(snapshot) {
    this.destroyCharts();

    const scopeLabel = snapshot.selectedSector || snapshot.selectedRow?.sector || 'Visible sectors';
    this.nodes.timelineCaption.textContent = snapshot.selectedSector
      ? `${snapshot.selectedSector} through the last ${this.state.days} days.`
      : 'Combined news and search pressure through the selected lookback window.';
    this.nodes.timelinePill.textContent = snapshot.selectedSector || `${snapshot.summary.visibleSectorCount} sectors`;

    this.charts.timeline = new window.Chart(this.nodes.timelineChart.getContext('2d'), {
      type: 'line',
      data: {
        labels: snapshot.series.dates.map((dateValue) => shortDateLabel(dateValue)),
        datasets: [
          {
            label: 'All-surface negative',
            data: snapshot.series.allSurfaceNeg,
            borderColor: '#ff8261',
            backgroundColor: 'rgba(255,130,97,.18)',
            fill: false,
            tension: 0.22,
          },
          {
            label: 'News negative',
            data: snapshot.series.newsNeg,
            borderColor: '#f59e0b',
            backgroundColor: 'rgba(245,158,11,.18)',
            fill: false,
            tension: 0.22,
          },
          {
            label: 'Organic negative',
            data: snapshot.series.organicNeg,
            borderColor: '#ff5e57',
            backgroundColor: 'rgba(255,94,87,.18)',
            fill: false,
            tension: 0.22,
          },
          {
            label: 'All-surface control',
            data: snapshot.series.allSurfaceCtrl,
            borderColor: '#58dbed',
            backgroundColor: 'rgba(88,219,237,.16)',
            fill: false,
            tension: 0.22,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { labels: { color: '#ebf2f2' } },
          tooltip: {
            callbacks: {
              label: (context) => `${context.dataset.label}: ${context.parsed.y.toFixed(1)}%`,
            },
          },
        },
        scales: {
          x: {
            ticks: { color: '#ebf2f2', maxTicksLimit: 10 },
            grid: { color: 'rgba(255,255,255,.08)' },
          },
          y: {
            beginAtZero: true,
            max: 100,
            ticks: {
              color: '#ebf2f2',
              callback: (value) => `${value}%`,
            },
            grid: { color: 'rgba(255,255,255,.08)' },
          },
        },
      },
    });

    this.nodes.newsCaption.textContent = `${scopeLabel} across ${this.state.days} days.`;
    this.nodes.newsPill.textContent = snapshot.activeDate || 'No date';
    this.charts.news = new window.Chart(this.nodes.newsChart.getContext('2d'), {
      type: 'bar',
      data: {
        labels: snapshot.series.dates.map((dateValue) => shortDateLabel(dateValue)),
        datasets: [
          { label: 'Positive', data: snapshot.series.newsPos, backgroundColor: '#82c618', stack: 'news' },
          { label: 'Neutral', data: snapshot.series.newsNeu, backgroundColor: '#cfdbdd', stack: 'news' },
          { label: 'Negative', data: snapshot.series.newsNeg, backgroundColor: '#ff8261', stack: 'news' },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: '#ebf2f2' } },
        },
        scales: {
          x: {
            stacked: true,
            ticks: { color: '#ebf2f2', maxTicksLimit: 8 },
            grid: { color: 'rgba(255,255,255,.08)' },
          },
          y: {
            stacked: true,
            beginAtZero: true,
            max: 100,
            ticks: {
              color: '#ebf2f2',
              callback: (value) => `${value}%`,
            },
            grid: { color: 'rgba(255,255,255,.08)' },
          },
        },
      },
    });

    this.nodes.featureCaption.textContent = `${scopeLabel} split by feature type.`;
    this.nodes.featurePill.textContent = snapshot.selectedSector || 'All sectors';
    this.charts.feature = new window.Chart(this.nodes.featureChart.getContext('2d'), {
      type: 'line',
      data: {
        labels: snapshot.featureSeries.dates.map((dateValue) => shortDateLabel(dateValue)),
        datasets: snapshot.featureSeries.order.map((feature, index) => ({
          label: FEATURE_LABELS[feature] || feature,
          data: snapshot.featureSeries.series[feature] || [],
          borderColor: FEATURE_COLORS[index % FEATURE_COLORS.length],
          backgroundColor: `${FEATURE_COLORS[index % FEATURE_COLORS.length]}33`,
          fill: true,
          tension: 0.2,
        })),
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { labels: { color: '#ebf2f2' } },
        },
        scales: {
          x: {
            ticks: { color: '#ebf2f2', maxTicksLimit: 8 },
            grid: { color: 'rgba(255,255,255,.08)' },
          },
          y: {
            beginAtZero: true,
            max: 100,
            ticks: {
              color: '#ebf2f2',
              callback: (value) => `${value}%`,
            },
            grid: { color: 'rgba(255,255,255,.08)' },
          },
        },
      },
    });

    this.nodes.controlCaption.textContent = `${scopeLabel} control share across the visible search surface.`;
    this.nodes.controlPill.textContent = snapshot.selectedSector || 'All sectors';
    this.charts.control = new window.Chart(this.nodes.controlChart.getContext('2d'), {
      type: 'line',
      data: {
        labels: snapshot.featureControlSeries.dates.map((dateValue) => shortDateLabel(dateValue)),
        datasets: snapshot.featureControlSeries.order.map((feature, index) => ({
          label: FEATURE_LABELS[feature] || feature,
          data: snapshot.featureControlSeries.series[feature] || [],
          borderColor: CONTROL_COLORS[index % CONTROL_COLORS.length],
          backgroundColor: `${CONTROL_COLORS[index % CONTROL_COLORS.length]}33`,
          fill: true,
          tension: 0.2,
        })),
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { labels: { color: '#ebf2f2' } },
        },
        scales: {
          x: {
            ticks: { color: '#ebf2f2', maxTicksLimit: 8 },
            grid: { color: 'rgba(255,255,255,.08)' },
          },
          y: {
            beginAtZero: true,
            max: 100,
            ticks: {
              color: '#ebf2f2',
              callback: (value) => `${value}%`,
            },
            grid: { color: 'rgba(255,255,255,.08)' },
          },
        },
      },
    });

    this.nodes.leaderboardCaption.textContent = `Highest combined sector pressure on ${snapshot.activeDate || 'the active date'}.`;
    this.nodes.leaderboardPill.textContent = `${snapshot.leaderboard.length} ranked`;
    this.charts.leaderboard = new window.Chart(this.nodes.leaderboardChart.getContext('2d'), {
      type: 'bar',
      data: {
        labels: snapshot.leaderboard.map((row) => row.sector),
        datasets: [
          {
            label: 'Signal score',
            data: snapshot.leaderboard.map((row) => Number((row.riskScore * 100).toFixed(1))),
            backgroundColor: snapshot.leaderboard.map((row) => row.riskTone === 'High' ? '#ff8261' : row.riskTone === 'Medium' ? '#f59e0b' : '#58dbed'),
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        indexAxis: 'y',
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (context) => `${context.parsed.x.toFixed(1)} signal`,
            },
          },
        },
        scales: {
          x: {
            beginAtZero: true,
            ticks: { color: '#ebf2f2' },
            grid: { color: 'rgba(255,255,255,.08)' },
          },
          y: {
            ticks: { color: '#ebf2f2' },
            grid: { display: false },
          },
        },
      },
    });
  }

  applySnapshot(snapshot, { updateCharts = true, replaceUrl = true } = {}) {
    this.snapshot = snapshot;
    this.state.date = snapshot.activeDate;
    this.state.selectedSector = snapshot.selectedSector;
    this.syncControls(snapshot);
    this.renderSummary(snapshot);
    this.renderSpotlight(snapshot);
    this.renderTable(snapshot);
    if (updateCharts) {
      this.renderCharts(snapshot);
    }
    this.updateUrl({ replace: replaceUrl });
    this.updateOpenLink();
    this.setFeedback(snapshot.summary.hint);
  }

  async load({ replaceUrl = true } = {}) {
    await this.store.ensureCore(this.state.days);
    const snapshot = this.store.buildSnapshot({
      days: this.state.days,
      date: this.state.date,
      query: this.state.query,
      sectorFilter: this.state.sectorFilter,
      companySizeFilter: this.state.companySizeFilter,
      crisisOnly: this.state.crisisOnly,
      selectedSector: this.state.selectedSector,
    });
    this.applySnapshot(snapshot, { updateCharts: true, replaceUrl });
    this.scheduleStockLoad();
  }

  async refreshView({ replaceUrl = true, updateCharts = true } = {}) {
    if (!this.snapshot) {
      await this.load({ replaceUrl });
      return;
    }
    const snapshot = this.store.buildSnapshot({
      days: this.state.days,
      date: this.state.date,
      query: this.state.query,
      sectorFilter: this.state.sectorFilter,
      companySizeFilter: this.state.companySizeFilter,
      crisisOnly: this.state.crisisOnly,
      selectedSector: this.state.selectedSector,
    });
    this.applySnapshot(snapshot, { updateCharts, replaceUrl });
  }

  scheduleStockLoad() {
    if (this.stockScheduled) return;
    this.stockScheduled = true;
    scheduleIdle(async () => {
      try {
        await this.store.ensureStockData();
        if (this.destroyed) return;
        await this.refreshView({ replaceUrl: false, updateCharts: false });
      } catch (error) {
        console.warn('Deferred sector stock load failed', error);
      }
    });
  }

  resizeCharts() {
    Object.values(this.charts).forEach((chart) => {
      try {
        chart.resize();
      } catch (_error) {
        // Ignore resize issues during hidden panel transitions.
      }
    });
  }

  async init() {
    this.readUrlState();
    this.render();
    this.bind();
    await ensureChartJs();
    await this.load({ replaceUrl: false });
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
        // Ignore cleanup failures on teardown.
      }
    });
    this.destroyCharts();
    this.host.innerHTML = '';
    this.host.hidden = true;
  }
}

export async function prefetchSectorTab({ tabId, shellConfig, getDirectUrl }) {
  const cacheKey = `${shellConfig.view || 'external'}:${tabId}`;
  if (prefetchRegistry.has(cacheKey)) return prefetchRegistry.get(cacheKey);

  const task = (async () => {
    const directUrl = getDirectUrl();
    const requestedDays = Number(directUrl.searchParams.get('days'));
    const days = LOOKBACK_OPTIONS.includes(requestedDays) ? requestedDays : 30;
    await ensureChartJs();
    await getSectorStore({
      view: shellConfig.view || 'external',
      tabId,
    }).ensureCore(days);
  })();

  prefetchRegistry.set(cacheKey, task);
  try {
    await task;
  } catch (error) {
    prefetchRegistry.delete(cacheKey);
    throw error;
  }
}

export async function mountSectorTab({ host, getDirectUrl, onHistoryChange, tabId, shellConfig }) {
  const controller = new SectorTabController({
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
