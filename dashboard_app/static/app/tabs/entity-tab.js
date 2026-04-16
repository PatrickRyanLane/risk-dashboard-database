import { ensureChartJs } from '../chart-runtime.js';
import { getEntityConfig, isNativeEntityTab as isNativeEntityTabConfig } from '../entity-config.js';
import { getEntityStore, computeCompositeSignal } from '../entity-store.js';
import { runInternalRefresh } from '../internal-refresh.js';
import { clearAllNativeDataStores } from '../native-stores.js';
import { enableCardDragReorder } from '../card-reorder.js';

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
  const percent = Math.abs(value) <= 1 ? value * 100 : value;
  return `${percent.toFixed(digits)}%`;
}

function formatPercentPoints(value, digits = 1, fallback = 'N/A') {
  if (value == null || Number.isNaN(value)) return fallback;
  return `${Number(value).toFixed(digits)}%`;
}

function formatSignedPercent(value) {
  if (value == null || Number.isNaN(value)) return 'Pending';
  const percent = Math.abs(value) <= 1 ? value * 100 : value;
  const prefix = percent > 0 ? '+' : '';
  return `${prefix}${percent.toFixed(1)}%`;
}

function formatInteger(value, fallback = '0') {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return fallback;
  return `${Math.round(numeric)}`;
}

function buildAutoPresetName(label = 'Preset') {
  const now = new Date();
  const year = now.getFullYear();
  const month = `${now.getMonth() + 1}`.padStart(2, '0');
  const day = `${now.getDate()}`.padStart(2, '0');
  const hours = `${now.getHours()}`.padStart(2, '0');
  const minutes = `${now.getMinutes()}`.padStart(2, '0');
  return `${label} ${year}-${month}-${day} ${hours}:${minutes}`;
}

function scoreTone(label) {
  if (label === 'High') return 'high';
  if (label === 'Medium') return 'medium';
  if (label === 'Low') return 'low';
  return 'neutral';
}

const HELP_TEXT = {
  news: 'Share of negative news articles over time. In index mode this aggregates all visible entities; when a row is selected it shows only that entity.',
  serp: 'Organic Page-1 SERP pressure over time: negative organic share and controlled organic share.',
  featureIndex: 'Index snapshot: percent of visible entities with each feature on the active date. Orange is entities with at least one negative URL in that feature; blue is entities with that feature and no negative URLs.',
  featureEntity: 'Selected entity snapshot: percent of URLs in each feature that are negative on the active date.',
  featureComposite: 'Stacked trend of negative share across page-one organic results and SERP features over time.',
  featurePresence: 'Tracks feature visibility over time across the lookback window. Use the metric toggle to switch between presence rate, slot share, negative share, and controlled share.',
  leaderboard: 'Weighted composite leaderboard. Date mode ranks active-date pressure; window mode ranks adjusted composite signal across the lookback range with volume-aware smoothing. In window mode, the basis toggle switches between All listings and Page-one only.',
};

const FEATURE_COMPOSITE_COLORS = [
  'rgba(255,130,97,.82)',
  'rgba(90,208,225,.8)',
  'rgba(130,198,22,.78)',
  'rgba(247,199,82,.78)',
  'rgba(170,140,255,.78)',
  'rgba(108,143,255,.78)',
];

const FEATURE_PRESENCE_METRICS = {
  presence_rate: {
    label: 'Presence rate',
    shortLabel: 'Presence',
    description: 'Brands with feature / active brands',
    countLabel: 'brands',
    percentScale: true,
  },
  slot_share: {
    label: 'Slot share',
    shortLabel: 'Slot share',
    description: 'Feature slots / all page-one slots',
    countLabel: 'slots',
    percentScale: true,
  },
  negative_share: {
    label: 'Negative share',
    shortLabel: 'Negative share',
    description: 'Negative slots / feature slots',
    countLabel: 'negative slots',
    percentScale: true,
  },
  controlled_share: {
    label: 'Controlled share',
    shortLabel: 'Controlled share',
    description: 'Controlled slots / feature slots',
    countLabel: 'controlled slots',
    percentScale: true,
  },
  serp_size: {
    label: 'Page-One Real Estate',
    shortLabel: 'Real Estate',
    description: 'Total page-one slots per day (organic + SERP features)',
    countLabel: 'slots',
    percentScale: false,
  },
  serp_size_stacked: {
    label: 'Page-One Real Estate (stacked)',
    shortLabel: 'Real Estate (stacked)',
    description: 'Total page-one slots per day, split by feature',
    countLabel: 'slots',
    percentScale: false,
  },
};

const FEATURE_TYPE_COLORS = {
  organic: 'rgba(255,130,97,.88)',
  aio_citations: 'rgba(90,208,225,.88)',
  paa_items: 'rgba(130,198,22,.88)',
  videos_items: 'rgba(247,199,82,.88)',
  perspectives_items: 'rgba(170,140,255,.88)',
  top_stories_items: 'rgba(108,143,255,.88)',
};

const LEADERBOARD_COMPONENTS = [
  { key: 'newsNegative', totalKey: 'newsTotal', weightKey: 'newsNegative', label: 'News', color: 'rgba(255,130,97,.78)' },
  { key: 'organicNegative', totalKey: 'organicTotal', weightKey: 'organicNegative', label: 'Organic', color: 'rgba(88,219,237,.78)' },
  { key: 'topStoriesNegative', totalKey: 'topStoriesTotal', weightKey: 'topStoriesNegative', label: 'Top stories', color: 'rgba(247,199,82,.78)' },
  { key: 'aioNegative', totalKey: 'aioTotal', weightKey: 'aioCitationsNegative', label: 'AIO', color: 'rgba(130,198,22,.78)' },
  { key: 'paaNegative', totalKey: 'paaTotal', weightKey: 'paaNegative', label: 'PAA', color: 'rgba(170,140,255,.78)' },
  { key: 'videosNegative', totalKey: 'videosTotal', weightKey: 'videosNegative', label: 'Videos', color: 'rgba(108,143,255,.78)' },
  { key: 'perspectivesNegative', totalKey: 'perspectivesTotal', weightKey: 'perspectivesNegative', label: 'Perspectives', color: 'rgba(201,226,120,.78)' },
];

const FEATURE_TYPE_LABELS = {
  organic: 'Organic',
  all_page_one_slots: 'All page-one slots',
  aio_citations: 'AIO citations',
  paa_items: 'PAA',
  videos_items: 'Videos',
  perspectives_items: 'Perspectives',
  top_stories_items: 'Top stories',
};

const FEATURE_MODAL_FILTER_ORDER = [
  'organic',
  'top_stories_items',
  'aio_citations',
  'paa_items',
  'videos_items',
  'perspectives_items',
];

const FEATURE_TYPE_ALIASES = {
  organic: 'organic',
  aio: 'aio_citations',
  'aio citation': 'aio_citations',
  'aio citations': 'aio_citations',
  'ai overview': 'aio_citations',
  'paa': 'paa_items',
  'paa item': 'paa_items',
  'paa items': 'paa_items',
  'people also ask': 'paa_items',
  videos: 'videos_items',
  'videos item': 'videos_items',
  'videos items': 'videos_items',
  perspectives: 'perspectives_items',
  'perspectives item': 'perspectives_items',
  'perspectives items': 'perspectives_items',
  'top stories': 'top_stories_items',
  'top stories item': 'top_stories_items',
  'top stories items': 'top_stories_items',
};

function normalizeFeatureType(value) {
  const raw = String(value || '').trim();
  if (!raw) return '';
  if (FEATURE_TYPE_LABELS[raw]) return raw;
  const normalized = raw.replace(/[_-]+/g, ' ').replace(/\s+/g, ' ').trim().toLowerCase();
  return FEATURE_TYPE_ALIASES[normalized] || '';
}

function featureTypeLabel(value) {
  const normalized = normalizeFeatureType(value);
  return FEATURE_TYPE_LABELS[normalized] || String(value || '').trim();
}

function normalizeOrganicModalRows(rows) {
  return (Array.isArray(rows) ? rows : []).map((row) => ({
    ...row,
    modal_kind: 'organic_serp',
    feature_type: 'organic',
    control_class: row.controlled,
    id: row.serp_result_id || row.id,
  }));
}

function toFinite(value, fallback = 0) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function normalizeSignalWeights(weights = {}, fallback = {}) {
  const featureFallbackSum = (
    (fallback.topStoriesNegative ?? 0)
    + (fallback.aioCitationsNegative ?? 0)
    + (fallback.paaNegative ?? 0)
    + (fallback.videosNegative ?? 0)
    + (fallback.perspectivesNegative ?? 0)
    || 1
  );
  const legacyFeatureWeight = toFinite(
    weights.serpNegativeFeatures,
    (
      (fallback.topStoriesNegative ?? 0.16)
      + (fallback.aioCitationsNegative ?? 0.12)
      + (fallback.paaNegative ?? 0.1)
      + (fallback.videosNegative ?? 0.07)
      + (fallback.perspectivesNegative ?? 0.07)
    ),
  );
  const featureScale = legacyFeatureWeight / featureFallbackSum;
  return {
    newsNegative: clamp(toFinite(weights.newsNegative, fallback.newsNegative ?? 0.24), 0, 1),
    organicNegative: clamp(
      toFinite(weights.organicNegative, toFinite(weights.serpNegativeOrganic, fallback.organicNegative ?? 0.24)),
      0,
      1,
    ),
    topStoriesNegative: clamp(
      toFinite(weights.topStoriesNegative, (fallback.topStoriesNegative ?? 0.16) * featureScale),
      0,
      1,
    ),
    aioCitationsNegative: clamp(
      toFinite(weights.aioCitationsNegative, (fallback.aioCitationsNegative ?? 0.12) * featureScale),
      0,
      1,
    ),
    paaNegative: clamp(
      toFinite(weights.paaNegative, (fallback.paaNegative ?? 0.1) * featureScale),
      0,
      1,
    ),
    videosNegative: clamp(
      toFinite(weights.videosNegative, (fallback.videosNegative ?? 0.07) * featureScale),
      0,
      1,
    ),
    perspectivesNegative: clamp(
      toFinite(weights.perspectivesNegative, (fallback.perspectivesNegative ?? 0.07) * featureScale),
      0,
      1,
    ),
    serpControl: clamp(toFinite(weights.serpControl, fallback.serpControl ?? 0.1), 0, 1),
  };
}

function formatWeightPercent(value) {
  return `${(toFinite(value, 0) * 100).toFixed(0)}%`;
}

function withAlpha(color, alpha = 1) {
  const safeAlpha = clamp(toFinite(alpha, 1), 0, 1);
  if (typeof color !== 'string') return color;
  const value = color.trim();
  const rgbaMatch = value.match(/^rgba?\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)(?:\s*,\s*([0-9.]+))?\s*\)$/i);
  if (rgbaMatch) {
    const r = Number(rgbaMatch[1]) || 0;
    const g = Number(rgbaMatch[2]) || 0;
    const b = Number(rgbaMatch[3]) || 0;
    return `rgba(${Math.max(0, Math.min(255, Math.round(r)))},${Math.max(0, Math.min(255, Math.round(g)))},${Math.max(0, Math.min(255, Math.round(b)))},${safeAlpha})`;
  }
  const hex = value.replace('#', '');
  if (/^[0-9a-f]{3}$/i.test(hex)) {
    const r = parseInt(hex[0] + hex[0], 16);
    const g = parseInt(hex[1] + hex[1], 16);
    const b = parseInt(hex[2] + hex[2], 16);
    return `rgba(${r},${g},${b},${safeAlpha})`;
  }
  if (/^[0-9a-f]{6}$/i.test(hex)) {
    const r = parseInt(hex.slice(0, 2), 16);
    const g = parseInt(hex.slice(2, 4), 16);
    const b = parseInt(hex.slice(4, 6), 16);
    return `rgba(${r},${g},${b},${safeAlpha})`;
  }
  return color;
}

function pearsonCorrelation(xs, ys) {
  const n = Math.min(xs.length, ys.length);
  if (n < 3) return null;
  const xMean = xs.slice(0, n).reduce((sum, value) => sum + value, 0) / n;
  const yMean = ys.slice(0, n).reduce((sum, value) => sum + value, 0) / n;
  let cov = 0;
  let xVar = 0;
  let yVar = 0;
  for (let index = 0; index < n; index += 1) {
    const xDelta = xs[index] - xMean;
    const yDelta = ys[index] - yMean;
    cov += xDelta * yDelta;
    xVar += xDelta * xDelta;
    yVar += yDelta * yDelta;
  }
  if (xVar <= 0 || yVar <= 0) return null;
  return cov / Math.sqrt(xVar * yVar);
}

function scheduleIdle(callback) {
  if (window.requestIdleCallback) {
    return window.requestIdleCallback(callback);
  }
  return window.setTimeout(callback, 0);
}

function createScaffold(config) {
  const lookbacks = config.lookbackOptions.map((days) => `
    <button type="button" class="entity-lookback" data-lookback="${days}" aria-pressed="false">${days}d</button>
  `).join('');

  return `
    <div class="entity-tab" data-entity-tab="${escapeHtml(config.tabId)}">
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
        <div class="entity-lookbacks" data-role="lookbacks">${lookbacks}</div>
        <label class="entity-field">
          <span>Date</span>
          <select data-role="date-select"></select>
        </label>
        <label class="entity-field" style="min-width:260px;flex:1 1 280px;">
          <span>Filter ${escapeHtml(config.label.toLowerCase())}</span>
          <input type="search" data-role="query-input" placeholder="Search by ${escapeHtml(config.label === 'Brands' ? 'brand, sector, or ticker' : 'CEO, company, or sector')}" />
        </label>
        <div class="entity-toolbar-actions">
          ${config.isInternal ? '<button type="button" class="entity-action" data-role="refresh-button">Refresh Data</button>' : ''}
          <button type="button" class="entity-action" data-role="calibrate-toggle">Calibration</button>
          <button type="button" class="entity-action entity-action--reset" data-role="reset-button">Reset View</button>
          <a class="entity-action" data-role="open-link" target="_blank" rel="noopener">Open Full Dashboard</a>
        </div>
      </section>

      <div class="entity-feedback" data-role="feedback"></div>

      <section class="entity-calibration" data-role="calibration-panel" hidden>
        <div class="entity-calibration-head">
          <p class="entity-kicker" style="margin:0;">Calibration Mode</p>
          <p class="entity-calibration-metric" data-role="calibration-metric">Awaiting data for calibration.</p>
        </div>
        <div class="entity-calibration-presets">
          <label class="entity-calibration-field">
            <span>Weight Preset</span>
            <select data-role="preset-select"></select>
          </label>
          <label class="entity-calibration-field">
            <span>Preset Name</span>
            <input type="text" data-role="preset-name" placeholder="Name this preset" maxlength="80" />
          </label>
          <button type="button" class="entity-action" data-role="preset-save">Save Preset</button>
          <button type="button" class="entity-action entity-action--reset" data-role="preset-delete">Delete Preset</button>
        </div>
        <div class="entity-calibration-grid">
          <label class="entity-calibration-field">
            <span>News Weight</span>
            <input type="range" min="0" max="1" step="0.01" data-role="weight-news" />
            <output data-role="weight-news-value">0%</output>
          </label>
          <label class="entity-calibration-field">
            <span>Organic SERP Weight</span>
            <input type="range" min="0" max="1" step="0.01" data-role="weight-serp" />
            <output data-role="weight-serp-value">0%</output>
          </label>
          <label class="entity-calibration-field">
            <span>Top Stories Weight</span>
            <input type="range" min="0" max="1" step="0.01" data-role="weight-top-stories" />
            <output data-role="weight-top-stories-value">0%</output>
          </label>
          <label class="entity-calibration-field">
            <span>AIO Citations Weight</span>
            <input type="range" min="0" max="1" step="0.01" data-role="weight-aio" />
            <output data-role="weight-aio-value">0%</output>
          </label>
          <label class="entity-calibration-field">
            <span>PAA Weight</span>
            <input type="range" min="0" max="1" step="0.01" data-role="weight-paa" />
            <output data-role="weight-paa-value">0%</output>
          </label>
          <label class="entity-calibration-field">
            <span>Videos Weight</span>
            <input type="range" min="0" max="1" step="0.01" data-role="weight-videos" />
            <output data-role="weight-videos-value">0%</output>
          </label>
          <label class="entity-calibration-field">
            <span>Perspectives Weight</span>
            <input type="range" min="0" max="1" step="0.01" data-role="weight-perspectives" />
            <output data-role="weight-perspectives-value">0%</output>
          </label>
          <label class="entity-calibration-field">
            <span>SERP Control Penalty</span>
            <input type="range" min="0" max="1" step="0.01" data-role="weight-control" />
            <output data-role="weight-control-value">0%</output>
          </label>
        </div>
        <div class="entity-calibration-actions">
          <button type="button" class="entity-action" data-role="calibration-auto">Auto-fit to downside stock</button>
          <button type="button" class="entity-action entity-action--reset" data-role="calibration-reset">Reset Weights</button>
        </div>
      </section>

      <section class="entity-grid">
        <div class="entity-grid-primary" data-card-group="entity-primary">
          <article class="entity-card" data-card-id="news-negativity">
            <header class="entity-card-header">
              <div>
                <div class="entity-title-row">
                  <h3>News Negativity</h3>
                  <button type="button" class="entity-help" data-role="news-help" aria-label="Explain News Negativity">?</button>
                </div>
                <p data-role="news-caption">Selected entities over the current lookback window.</p>
              </div>
              <span class="entity-pill" data-role="news-pill">Awaiting data</span>
            </header>
            <div class="entity-chart-wrap"><canvas data-role="news-chart"></canvas></div>
          </article>

          <article class="entity-card" data-card-id="serp-feature-snapshot">
            <header class="entity-card-header">
              <div>
                <div class="entity-title-row">
                  <h3 data-role="feature-title">Negative Page One Snapshot</h3>
                  <button type="button" class="entity-help" data-role="feature-help" aria-label="Explain Negative Page One Snapshot">?</button>
                </div>
              </div>
              <span class="entity-pill" data-role="feature-pill">Awaiting data</span>
            </header>
            <div class="entity-chart-wrap"><canvas data-role="feature-chart"></canvas></div>
          </article>
        </div>

        <div class="entity-grid-secondary" data-card-group="entity-secondary">
          <article class="entity-card" data-card-id="organic-search-results">
            <header class="entity-card-header">
              <div>
                <div class="entity-title-row">
                  <h3 data-role="serp-title">Organic Search Results</h3>
                  <button type="button" class="entity-help" data-role="serp-help" aria-label="Explain Organic Search Results">?</button>
                </div>
              </div>
              <span class="entity-pill" data-role="serp-pill">Awaiting data</span>
            </header>
            <div class="entity-chart-wrap"><canvas data-role="serp-chart"></canvas></div>
          </article>

          <article class="entity-card" data-card-id="negative-serp-feature-composite">
            <header class="entity-card-header">
              <div>
                <div class="entity-title-row">
                  <h3 data-role="feature-composite-title">Negative Page One SERP Composite</h3>
                  <button type="button" class="entity-help" data-role="feature-composite-help" aria-label="Explain Negative Page One SERP Composite">?</button>
                </div>
              </div>
              <div class="entity-card-tools">
                <div class="entity-view-toggle" role="group" aria-label="Feature composite chart type">
                  <button type="button" class="entity-view-toggle-btn" data-role="feature-composite-view" data-view="bar" aria-pressed="true">Bars</button>
                  <button type="button" class="entity-view-toggle-btn" data-role="feature-composite-view" data-view="area" aria-pressed="false">Area</button>
                </div>
                <span class="entity-pill" data-role="feature-composite-pill">Awaiting data</span>
              </div>
            </header>
            <div class="entity-chart-wrap"><canvas data-role="feature-composite-chart"></canvas></div>
          </article>

          <article class="entity-card" data-card-id="serp-feature-trends">
            <header class="entity-card-header">
              <div>
                <div class="entity-title-row">
                  <h3 data-role="feature-presence-title">SERP Feature Trends</h3>
                  <button type="button" class="entity-help" data-role="feature-presence-help" aria-label="Explain SERP Feature Trends">?</button>
                </div>
                <p data-role="feature-presence-caption">Presence rate over the current lookback window.</p>
              </div>
              <div class="entity-card-tools">
                <label class="entity-mini-field">
                  <span>Metric</span>
                  <select data-role="feature-presence-metric" aria-label="SERP feature trend metric">
                    <option value="presence_rate">Presence</option>
                    <option value="slot_share">Slot share</option>
                    <option value="negative_share">Negative share</option>
                    <option value="controlled_share">Controlled share</option>
                    <option value="serp_size">Page-One Real Estate</option>
                    <option value="serp_size_stacked">Page-One Real Estate (stacked)</option>
                  </select>
                </label>
                <span class="entity-pill" data-role="feature-presence-pill">Awaiting data</span>
              </div>
            </header>
            <div class="entity-chart-wrap"><canvas data-role="feature-presence-chart"></canvas></div>
          </article>

          <article class="entity-card" data-card-id="signal-leaderboard">
            <header class="entity-card-header">
              <div>
                <div class="entity-title-row">
                  <h3>Signal Leaderboard</h3>
                  <button type="button" class="entity-help" data-role="leaderboard-help" aria-label="Explain Signal Leaderboard">?</button>
                </div>
                <p data-role="leaderboard-caption">Highest combined crisis pressure on the active date.</p>
              </div>
              <div class="entity-card-tools">
                <div class="entity-view-toggle" role="group" aria-label="Leaderboard scope">
                  <button type="button" class="entity-view-toggle-btn" data-role="leaderboard-scope" data-scope="date" aria-pressed="true">Date</button>
                  <button type="button" class="entity-view-toggle-btn" data-role="leaderboard-scope" data-scope="window" aria-pressed="false">Window</button>
                </div>
                <div class="entity-view-toggle" role="group" aria-label="Window leaderboard direction">
                  <button type="button" class="entity-view-toggle-btn" data-role="leaderboard-direction" data-direction="worst" aria-pressed="true">Worst</button>
                  <button type="button" class="entity-view-toggle-btn" data-role="leaderboard-direction" data-direction="best" aria-pressed="false">Best</button>
                </div>
                <label class="entity-mini-field">
                  <span>Min/day</span>
                  <select data-role="leaderboard-min-volume" aria-label="Minimum listings per active day">
                    <option value="0">Any</option>
                    <option value="2">2+</option>
                    <option value="5">5+</option>
                    <option value="10">10+</option>
                    <option value="20">20+</option>
                  </select>
                </label>
                <label class="entity-mini-field">
                  <span>Basis</span>
                  <select data-role="leaderboard-listing-basis" aria-label="Window leaderboard listing basis">
                    <option value="all">All</option>
                    <option value="page_one">Page one</option>
                  </select>
                </label>
                <span class="entity-pill" data-role="leaderboard-pill">Awaiting data</span>
              </div>
            </header>
            <div class="entity-chart-wrap"><canvas data-role="leaderboard-chart"></canvas></div>
          </article>
        </div>
      </section>

      <section class="entity-table-card">
        <div class="entity-table-head">
          <div>
            <h3>${escapeHtml(config.label)} Watchlist</h3>
            <p data-role="table-caption">Rows update immediately from native shell state.</p>
          </div>
          <span class="entity-pill" data-role="table-pill">0 rows</span>
        </div>
        <div class="entity-table-scroll">
          <table class="entity-table">
            <thead data-role="table-head"></thead>
            <tbody data-role="table-body"></tbody>
          </table>
        </div>
      </section>

      <div class="entity-modal" data-role="entity-modal" hidden>
        <div class="entity-modal-backdrop" data-role="entity-modal-backdrop"></div>
        <div class="entity-modal-card" role="dialog" aria-modal="true" aria-label="Entity details">
          <div class="entity-modal-head">
            <h3 data-role="entity-modal-title">Entity Details</h3>
            <button type="button" class="entity-action entity-action--reset" data-role="entity-modal-close">Close</button>
          </div>
          <div class="entity-modal-body" data-role="entity-modal-body"></div>
          <div class="entity-modal-actions">
            <button type="button" class="entity-action" data-role="entity-modal-focus">Focus Entity</button>
            <a class="entity-action" data-role="entity-modal-open-link" target="_blank" rel="noopener">Open Full Dashboard</a>
          </div>
        </div>
      </div>
    </div>
  `;
}

class EntityTabController {
  constructor({ host, tabId, shellConfig, getDirectUrl, onHistoryChange }) {
    this.host = host;
    this.tabId = tabId;
    this.shellConfig = shellConfig;
    this.getDirectUrl = getDirectUrl;
    this.onHistoryChange = onHistoryChange;
    this.config = getEntityConfig(tabId, shellConfig);
    this.store = getEntityStore(this.config);
    this.signalWeights = normalizeSignalWeights(this.config.signalSettings?.weights, this.config.signalSettings?.weights);
    this.savedPresets = {};
    this.activePresetKey = '__default__';
    this.sharedPresetsEnabled = false;
    this.modalEntity = null;
    this.modalState = null;
    this.modalLoadToken = 0;
    this.cleanups = [];
    this.charts = {};
    this.externalTooltipIds = new Set();
    this.loadToken = 0;
    this.destroyed = false;
    this.supplementalLoadScheduled = false;
    this.rows = [];
    this.core = null;
    this.legendTriStates = {};
    this.state = {
      days: this.config.defaultDays,
      date: '',
      query: '',
      selectedEntity: '',
      sortKey: 'riskScore',
      sortDir: 'desc',
      calibrationOpen: false,
      featureCompositeView: 'bar',
      featurePresenceMetric: 'presence_rate',
      leaderboardScope: 'date',
      leaderboardDirection: 'worst',
      leaderboardMinDailyListings: 0,
      leaderboardListingBasis: 'all',
    };
  }

  render() {
    this.host.innerHTML = createScaffold(this.config);
    this.root = this.host.firstElementChild;
    this.nodes = {
      lookbacks: Array.from(this.root.querySelectorAll('[data-lookback]')),
      dateSelect: this.root.querySelector('[data-role="date-select"]'),
      queryInput: this.root.querySelector('[data-role="query-input"]'),
      resetButton: this.root.querySelector('[data-role="reset-button"]'),
      calibrateToggle: this.root.querySelector('[data-role="calibrate-toggle"]'),
      refreshButton: this.root.querySelector('[data-role="refresh-button"]'),
      openLink: this.root.querySelector('[data-role="open-link"]'),
      feedback: this.root.querySelector('[data-role="feedback"]'),
      calibrationPanel: this.root.querySelector('[data-role="calibration-panel"]'),
      calibrationMetric: this.root.querySelector('[data-role="calibration-metric"]'),
      calibrationAuto: this.root.querySelector('[data-role="calibration-auto"]'),
      calibrationReset: this.root.querySelector('[data-role="calibration-reset"]'),
      presetSelect: this.root.querySelector('[data-role="preset-select"]'),
      presetName: this.root.querySelector('[data-role="preset-name"]'),
      presetSave: this.root.querySelector('[data-role="preset-save"]'),
      presetDelete: this.root.querySelector('[data-role="preset-delete"]'),
      weightNews: this.root.querySelector('[data-role="weight-news"]'),
      weightSerp: this.root.querySelector('[data-role="weight-serp"]'),
      weightTopStories: this.root.querySelector('[data-role="weight-top-stories"]'),
      weightAio: this.root.querySelector('[data-role="weight-aio"]'),
      weightPaa: this.root.querySelector('[data-role="weight-paa"]'),
      weightVideos: this.root.querySelector('[data-role="weight-videos"]'),
      weightPerspectives: this.root.querySelector('[data-role="weight-perspectives"]'),
      weightControl: this.root.querySelector('[data-role="weight-control"]'),
      weightNewsValue: this.root.querySelector('[data-role="weight-news-value"]'),
      weightSerpValue: this.root.querySelector('[data-role="weight-serp-value"]'),
      weightTopStoriesValue: this.root.querySelector('[data-role="weight-top-stories-value"]'),
      weightAioValue: this.root.querySelector('[data-role="weight-aio-value"]'),
      weightPaaValue: this.root.querySelector('[data-role="weight-paa-value"]'),
      weightVideosValue: this.root.querySelector('[data-role="weight-videos-value"]'),
      weightPerspectivesValue: this.root.querySelector('[data-role="weight-perspectives-value"]'),
      weightControlValue: this.root.querySelector('[data-role="weight-control-value"]'),
      summaryGrid: this.root.querySelector('[data-role="summary-grid"]'),
      selectedSpotlight: this.root.querySelector('[data-role="selected-spotlight"]'),
      tableHead: this.root.querySelector('[data-role="table-head"]'),
      tableBody: this.root.querySelector('[data-role="table-body"]'),
      newsCaption: this.root.querySelector('[data-role="news-caption"]'),
      newsPill: this.root.querySelector('[data-role="news-pill"]'),
      serpTitle: this.root.querySelector('[data-role="serp-title"]'),
      serpPill: this.root.querySelector('[data-role="serp-pill"]'),
      featureTitle: this.root.querySelector('[data-role="feature-title"]'),
      featurePill: this.root.querySelector('[data-role="feature-pill"]'),
      featureCompositeTitle: this.root.querySelector('[data-role="feature-composite-title"]'),
      featureCompositePill: this.root.querySelector('[data-role="feature-composite-pill"]'),
      featureCompositeViewButtons: Array.from(this.root.querySelectorAll('[data-role="feature-composite-view"]')),
      featurePresenceTitle: this.root.querySelector('[data-role="feature-presence-title"]'),
      featurePresenceCaption: this.root.querySelector('[data-role="feature-presence-caption"]'),
      featurePresenceMetric: this.root.querySelector('[data-role="feature-presence-metric"]'),
      featurePresencePill: this.root.querySelector('[data-role="feature-presence-pill"]'),
      leaderboardPill: this.root.querySelector('[data-role="leaderboard-pill"]'),
      leaderboardCaption: this.root.querySelector('[data-role="leaderboard-caption"]'),
      leaderboardScopeButtons: Array.from(this.root.querySelectorAll('[data-role="leaderboard-scope"]')),
      leaderboardDirectionButtons: Array.from(this.root.querySelectorAll('[data-role="leaderboard-direction"]')),
      leaderboardMinVolume: this.root.querySelector('[data-role="leaderboard-min-volume"]'),
      leaderboardListingBasis: this.root.querySelector('[data-role="leaderboard-listing-basis"]'),
      newsHelp: this.root.querySelector('[data-role="news-help"]'),
      featureHelp: this.root.querySelector('[data-role="feature-help"]'),
      featureCompositeHelp: this.root.querySelector('[data-role="feature-composite-help"]'),
      featurePresenceHelp: this.root.querySelector('[data-role="feature-presence-help"]'),
      serpHelp: this.root.querySelector('[data-role="serp-help"]'),
      leaderboardHelp: this.root.querySelector('[data-role="leaderboard-help"]'),
      tableCaption: this.root.querySelector('[data-role="table-caption"]'),
      tablePill: this.root.querySelector('[data-role="table-pill"]'),
      newsChart: this.root.querySelector('[data-role="news-chart"]'),
      serpChart: this.root.querySelector('[data-role="serp-chart"]'),
      featureChart: this.root.querySelector('[data-role="feature-chart"]'),
      featureCompositeChart: this.root.querySelector('[data-role="feature-composite-chart"]'),
      featurePresenceChart: this.root.querySelector('[data-role="feature-presence-chart"]'),
      leaderboardChart: this.root.querySelector('[data-role="leaderboard-chart"]'),
      entityModal: this.root.querySelector('[data-role="entity-modal"]'),
      entityModalBackdrop: this.root.querySelector('[data-role="entity-modal-backdrop"]'),
      entityModalTitle: this.root.querySelector('[data-role="entity-modal-title"]'),
      entityModalBody: this.root.querySelector('[data-role="entity-modal-body"]'),
      entityModalClose: this.root.querySelector('[data-role="entity-modal-close"]'),
      entityModalFocus: this.root.querySelector('[data-role="entity-modal-focus"]'),
      entityModalOpenLink: this.root.querySelector('[data-role="entity-modal-open-link"]'),
    };
    this.cleanups.push(...enableCardDragReorder({
      root: this.root,
      storageNamespace: `${this.config.view}:${this.config.tabId}`,
      onReorder: () => this.resizeCharts(),
    }));
    this.loadFeatureCompositeView();
    this.syncFeatureCompositeViewControls();
    this.syncFeaturePresenceMetricControl();
    this.loadPresets();
    this.renderPresetOptions(this.activePresetKey);
    this.syncCalibrationControls();
    this.setCalibrationOpen(this.state.calibrationOpen);
    this.syncLeaderboardControls();
    this.updateHelpText();
  }

  presetStorageKey() {
    return `riskdash.signalPresets:${this.config.view}:${this.config.tabId}`;
  }

  featureCompositeViewStorageKey() {
    return `riskdash.featureCompositeView:${this.config.view}:${this.config.tabId}`;
  }

  loadFeatureCompositeView() {
    try {
      const raw = String(localStorage.getItem(this.featureCompositeViewStorageKey()) || '').trim().toLowerCase();
      if (raw === 'bar' || raw === 'area') this.state.featureCompositeView = raw;
    } catch (_error) {
      this.state.featureCompositeView = 'bar';
    }
  }

  persistFeatureCompositeView() {
    try {
      localStorage.setItem(this.featureCompositeViewStorageKey(), this.state.featureCompositeView);
    } catch (_error) {
      // Ignore storage failures.
    }
  }

  syncFeatureCompositeViewControls() {
    const nextView = this.state.featureCompositeView === 'area' ? 'area' : 'bar';
    if (!Array.isArray(this.nodes.featureCompositeViewButtons)) return;
    this.nodes.featureCompositeViewButtons.forEach((button) => {
      const buttonView = String(button.getAttribute('data-view') || '').trim().toLowerCase();
      const isActive = buttonView === nextView;
      button.setAttribute('aria-pressed', isActive ? 'true' : 'false');
      if (isActive) button.classList.add('active');
      else button.classList.remove('active');
    });
  }

  syncFeaturePresenceMetricControl() {
    if (!this.nodes.featurePresenceMetric) return;
    const supported = new Set(Object.keys(FEATURE_PRESENCE_METRICS));
    const nextMetric = supported.has(String(this.state.featurePresenceMetric || '').trim())
      ? String(this.state.featurePresenceMetric || '').trim()
      : 'presence_rate';
    this.state.featurePresenceMetric = nextMetric;
    this.nodes.featurePresenceMetric.value = nextMetric;
  }

  getLegendTriStateBucket(chartKey) {
    if (!this.legendTriStates[chartKey] || typeof this.legendTriStates[chartKey] !== 'object') {
      this.legendTriStates[chartKey] = {};
    }
    return this.legendTriStates[chartKey];
  }

  getLegendDatasetKey(dataset, datasetIndex) {
    const rawFeature = String(dataset?.rawFeature || '').trim();
    if (rawFeature) return rawFeature;
    const label = String(dataset?.label || '').trim();
    if (label) return label;
    return `dataset_${datasetIndex}`;
  }

  applyLegendTriStateStyles(chartKey, chart, { update = true } = {}) {
    if (!chart?.data?.datasets) return;
    const stateBucket = this.getLegendTriStateBucket(chartKey);
    const datasets = chart.data.datasets;
    const hasHighlight = datasets.some((dataset, index) => {
      const key = this.getLegendDatasetKey(dataset, index);
      return stateBucket[key] === 'highlight';
    });

    datasets.forEach((dataset, index) => {
      const key = this.getLegendDatasetKey(dataset, index);
      const state = stateBucket[key] || 'normal';
      const meta = chart.getDatasetMeta(index);
      if (!dataset.__legendBase) {
        dataset.__legendBase = {
          borderWidth: toFinite(dataset.borderWidth, 2),
          pointRadius: toFinite(dataset.pointRadius, 0),
          pointHoverRadius: toFinite(dataset.pointHoverRadius, 3),
          borderColor: dataset.borderColor,
          backgroundColor: dataset.backgroundColor,
          tension: dataset.tension,
          fill: dataset.fill,
        };
      }
      const base = dataset.__legendBase;

      if (state === 'hidden') {
        meta.hidden = true;
        return;
      }
      meta.hidden = false;

      if (hasHighlight) {
        if (state === 'highlight') {
          dataset.borderWidth = Math.max(3, toFinite(base.borderWidth, 2) + 1);
          dataset.borderColor = withAlpha(base.borderColor, 1);
          dataset.backgroundColor = withAlpha(base.backgroundColor, 0.95);
          dataset.pointRadius = Math.max(2, toFinite(base.pointRadius, 0));
          dataset.pointHoverRadius = Math.max(4, toFinite(base.pointHoverRadius, 3));
        } else {
          dataset.borderWidth = Math.max(1, toFinite(base.borderWidth, 2) - 1);
          dataset.borderColor = withAlpha(base.borderColor, 0.24);
          dataset.backgroundColor = withAlpha(base.backgroundColor, 0.2);
          dataset.pointRadius = 0;
          dataset.pointHoverRadius = 2;
        }
      } else {
        dataset.borderWidth = base.borderWidth;
        dataset.borderColor = base.borderColor;
        dataset.backgroundColor = base.backgroundColor;
        dataset.pointRadius = base.pointRadius;
        dataset.pointHoverRadius = base.pointHoverRadius;
      }
      dataset.tension = base.tension;
      dataset.fill = base.fill;
    });

    if (update) chart.update();
  }

  cycleLegendTriState(chartKey, chart, legendItem) {
    const datasetIndex = Number(legendItem?.datasetIndex);
    if (!Number.isInteger(datasetIndex) || datasetIndex < 0) return;
    const dataset = chart?.data?.datasets?.[datasetIndex];
    if (!dataset) return;
    const key = this.getLegendDatasetKey(dataset, datasetIndex);
    const stateBucket = this.getLegendTriStateBucket(chartKey);
    const currentState = stateBucket[key] || 'normal';
    const nextState = currentState === 'normal'
      ? 'highlight'
      : currentState === 'highlight'
        ? 'hidden'
        : 'normal';
    if (nextState === 'highlight') {
      Object.keys(stateBucket).forEach((bucketKey) => {
        if (bucketKey !== key && stateBucket[bucketKey] === 'highlight') {
          stateBucket[bucketKey] = 'normal';
        }
      });
    }
    stateBucket[key] = nextState;
    this.applyLegendTriStateStyles(chartKey, chart, { update: true });
  }

  buildLegendTriStateOptions(chartKey, legend = {}) {
    const labels = legend?.labels && typeof legend.labels === 'object' ? legend.labels : {};
    return {
      ...legend,
      labels: { ...labels },
      onClick: (_event, legendItem, legendContext) => {
        const chart = legendContext?.chart;
        if (!chart) return;
        this.cycleLegendTriState(chartKey, chart, legendItem);
      },
    };
  }

  hasActiveLegendTriState(chartKey) {
    const bucket = this.legendTriStates?.[chartKey];
    if (!bucket || typeof bucket !== 'object') return false;
    return Object.values(bucket).some((state) => state && state !== 'normal');
  }

  getOrCreateExternalChartTooltip(chartKey) {
    const safeKey = String(chartKey || 'default').replace(/[^a-z0-9_-]+/gi, '-').toLowerCase();
    const id = `entity-chart-tooltip-${this.config.tabId}-${safeKey}`;
    this.externalTooltipIds.add(id);
    let tooltipEl = document.getElementById(id);
    if (tooltipEl) return tooltipEl;
    tooltipEl = document.createElement('div');
    tooltipEl.id = id;
    tooltipEl.className = 'entity-chart-tooltip';
    tooltipEl.style.opacity = '0';
    tooltipEl.style.pointerEvents = 'none';
    document.body.appendChild(tooltipEl);
    return tooltipEl;
  }

  renderExternalChartTooltip(context, { chartKey = 'default' } = {}) {
    const tooltipEl = this.getOrCreateExternalChartTooltip(chartKey);
    const tooltipModel = context?.tooltip;
    const chart = context?.chart;
    if (!tooltipEl || !tooltipModel || !chart) return;

    if (!tooltipModel.opacity) {
      tooltipEl.style.opacity = '0';
      tooltipEl.style.pointerEvents = 'none';
      return;
    }

    const titleLines = Array.isArray(tooltipModel.title) ? tooltipModel.title : [];
    const bodyItems = Array.isArray(tooltipModel.body) ? tooltipModel.body : [];
    const footerLines = Array.isArray(tooltipModel.footer) ? tooltipModel.footer : [];
    const labelColors = Array.isArray(tooltipModel.labelColors) ? tooltipModel.labelColors : [];

    const html = [];
    if (titleLines.length) {
      html.push('<div class="entity-chart-tooltip-title">');
      titleLines.forEach((line) => {
        html.push(`<div>${escapeHtml(line)}</div>`);
      });
      html.push('</div>');
    }

    bodyItems.forEach((bodyItem, itemIndex) => {
      const lines = Array.isArray(bodyItem?.lines) ? bodyItem.lines : [];
      if (!lines.length) return;
      const colors = labelColors[itemIndex] || {};
      const border = colors.borderColor || 'rgba(162,235,243,.7)';
      const fill = colors.backgroundColor || 'rgba(162,235,243,.45)';
      html.push('<div class="entity-chart-tooltip-item">');
      lines.forEach((line, lineIndex) => {
        if (lineIndex === 0) {
          html.push(
            `<div class="entity-chart-tooltip-line">`
            + `<span class="entity-chart-tooltip-swatch" style="border-color:${escapeHtml(border)};background:${escapeHtml(fill)}"></span>`
            + `<span>${escapeHtml(line)}</span>`
            + '</div>',
          );
        } else {
          html.push(`<div class="entity-chart-tooltip-subline">${escapeHtml(line)}</div>`);
        }
      });
      html.push('</div>');
    });

    if (footerLines.length) {
      html.push('<div class="entity-chart-tooltip-footer">');
      footerLines.forEach((line) => {
        html.push(`<div>${escapeHtml(line)}</div>`);
      });
      html.push('</div>');
    }

    tooltipEl.innerHTML = html.join('');
    tooltipEl.style.opacity = '1';
    tooltipEl.style.pointerEvents = 'none';
    tooltipEl.style.maxHeight = `${Math.max(220, Math.round(window.innerHeight * 0.62))}px`;

    const canvasRect = chart.canvas.getBoundingClientRect();
    const caretX = Number(tooltipModel.caretX) || 0;
    const caretY = Number(tooltipModel.caretY) || 0;
    const gap = 14;

    let left = canvasRect.left + caretX + gap;
    let top = canvasRect.top + caretY + gap;

    const tooltipWidth = tooltipEl.offsetWidth;
    const tooltipHeight = tooltipEl.offsetHeight;
    const maxLeft = window.innerWidth - tooltipWidth - 8;
    const maxTop = window.innerHeight - tooltipHeight - 8;

    if (left > maxLeft) {
      left = canvasRect.left + caretX - tooltipWidth - gap;
    }
    if (top > maxTop) {
      top = maxTop;
    }

    left = Math.max(8, Math.min(left, maxLeft));
    top = Math.max(8, Math.min(top, maxTop));

    tooltipEl.style.left = `${Math.round(left)}px`;
    tooltipEl.style.top = `${Math.round(top)}px`;
  }

  syncLeaderboardControls() {
    const scope = this.state.leaderboardScope === 'window' ? 'window' : 'date';
    const direction = this.state.leaderboardDirection === 'best' ? 'best' : 'worst';
    const minDaily = Math.max(0, Number(this.state.leaderboardMinDailyListings) || 0);
    const listingBasis = this.state.leaderboardListingBasis === 'page_one' ? 'page_one' : 'all';

    if (Array.isArray(this.nodes.leaderboardScopeButtons)) {
      this.nodes.leaderboardScopeButtons.forEach((button) => {
        const value = String(button.getAttribute('data-scope') || '').trim().toLowerCase();
        const isActive = value === scope;
        button.setAttribute('aria-pressed', isActive ? 'true' : 'false');
        if (isActive) button.classList.add('active');
        else button.classList.remove('active');
      });
    }

    if (Array.isArray(this.nodes.leaderboardDirectionButtons)) {
      const isWindow = scope === 'window';
      this.nodes.leaderboardDirectionButtons.forEach((button) => {
        const value = String(button.getAttribute('data-direction') || '').trim().toLowerCase();
        const isActive = value === direction;
        button.setAttribute('aria-pressed', isActive ? 'true' : 'false');
        if (isActive) button.classList.add('active');
        else button.classList.remove('active');
        button.disabled = !isWindow;
      });
    }

    if (this.nodes.leaderboardMinVolume) {
      const supported = new Set(['0', '2', '5', '10', '20']);
      const minDailyValue = supported.has(String(minDaily)) ? String(minDaily) : '0';
      this.nodes.leaderboardMinVolume.value = minDailyValue;
      this.nodes.leaderboardMinVolume.disabled = scope !== 'window';
    }
    if (this.nodes.leaderboardListingBasis) {
      this.nodes.leaderboardListingBasis.value = listingBasis;
      this.nodes.leaderboardListingBasis.disabled = scope !== 'window';
    }
  }

  loadPresets() {
    try {
      const raw = localStorage.getItem(this.presetStorageKey());
      const parsed = raw ? JSON.parse(raw) : {};
      if (parsed && typeof parsed === 'object') {
        this.savedPresets = Object.fromEntries(
          Object.entries(parsed)
            .filter(([key, value]) => key && value && typeof value === 'object')
            .map(([key, value]) => [key, normalizeSignalWeights(value, this.config.signalSettings?.weights)]),
        );
      }
    } catch (_error) {
      this.savedPresets = {};
    }
  }

  sharedPresetEndpoint() {
    return `/api/internal/signal_presets?tab_id=${encodeURIComponent(this.config.tabId)}`;
  }

  async readResponseError(response, fallback = '') {
    let detail = fallback || `HTTP ${response.status}`;
    try {
      const payload = await response.json();
      detail = payload?.error || payload?.detail || detail;
    } catch (_error) {
      try {
        detail = await response.text() || detail;
      } catch (_error2) {
        // Ignore parse failures.
      }
    }
    return detail || `HTTP ${response.status}`;
  }

  async loadSharedPresets({ quiet = true } = {}) {
    if (!this.config.isInternal) return false;
    try {
      const response = await fetch(this.sharedPresetEndpoint(), { credentials: 'same-origin' });
      if (!response.ok) {
        throw new Error(await this.readResponseError(response, `HTTP ${response.status}`));
      }
      const payload = await response.json();
      const rows = Array.isArray(payload?.rows) ? payload.rows : [];
      this.savedPresets = Object.fromEntries(
        rows
          .map((row) => {
            const name = String(row?.preset_name || '').trim();
            if (!name) return null;
            return [name, normalizeSignalWeights(row?.weights || {}, this.config.signalSettings?.weights)];
          })
          .filter(Boolean),
      );
      this.persistPresets();
      this.sharedPresetsEnabled = true;
      this.renderPresetOptions(this.activePresetKey);
      return true;
    } catch (error) {
      this.sharedPresetsEnabled = false;
      if (!quiet) this.setFeedback(error?.message || 'Failed to load shared presets.', 'error');
      return false;
    }
  }

  async saveSharedPreset(presetName, weights) {
    const response = await fetch('/api/internal/signal_presets', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify({
        tab_id: this.config.tabId,
        preset_name: presetName,
        weights,
      }),
    });
    if (!response.ok) {
      throw new Error(await this.readResponseError(response, `HTTP ${response.status}`));
    }
    return response.json();
  }

  async deleteSharedPreset(presetName) {
    const response = await fetch('/api/internal/signal_presets', {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify({
        tab_id: this.config.tabId,
        preset_name: presetName,
      }),
    });
    if (!response.ok) {
      throw new Error(await this.readResponseError(response, `HTTP ${response.status}`));
    }
    return response.json();
  }

  persistPresets() {
    try {
      localStorage.setItem(this.presetStorageKey(), JSON.stringify(this.savedPresets));
    } catch (_error) {
      // Ignore storage failures.
    }
  }

  renderPresetOptions(selectedKey = '__default__') {
    if (!this.nodes.presetSelect) return;
    const keys = Object.keys(this.savedPresets).sort((a, b) => a.localeCompare(b));
    this.nodes.presetSelect.innerHTML = [
      '<option value="__default__">Default</option>',
      ...keys.map((key) => `<option value="${escapeHtml(key)}">${escapeHtml(key)}</option>`),
    ].join('');
    this.nodes.presetSelect.value = keys.includes(selectedKey) || selectedKey === '__default__'
      ? selectedKey
      : '__default__';
    this.activePresetKey = this.nodes.presetSelect.value;
    if (this.nodes.presetDelete) {
      this.nodes.presetDelete.disabled = this.activePresetKey === '__default__';
    }
    if (this.nodes.presetName) {
      this.nodes.presetName.value = this.activePresetKey === '__default__' ? '' : this.activePresetKey;
    }
  }

  updateHelpText(selectedEntity = '') {
    if (this.nodes.newsHelp) {
      this.nodes.newsHelp.title = HELP_TEXT.news;
      this.nodes.newsHelp.dataset.help = HELP_TEXT.news;
    }
    if (this.nodes.serpHelp) {
      this.nodes.serpHelp.title = HELP_TEXT.serp;
      this.nodes.serpHelp.dataset.help = HELP_TEXT.serp;
    }
    if (this.nodes.featureHelp) {
      const featureHelpText = selectedEntity ? HELP_TEXT.featureEntity : HELP_TEXT.featureIndex;
      this.nodes.featureHelp.title = featureHelpText;
      this.nodes.featureHelp.dataset.help = featureHelpText;
    }
    if (this.nodes.featureCompositeHelp) {
      this.nodes.featureCompositeHelp.title = HELP_TEXT.featureComposite;
      this.nodes.featureCompositeHelp.dataset.help = HELP_TEXT.featureComposite;
    }
    if (this.nodes.featurePresenceHelp) {
      this.nodes.featurePresenceHelp.title = this.buildFeaturePresenceHelpText();
      this.nodes.featurePresenceHelp.dataset.help = this.buildFeaturePresenceHelpText();
    }
    if (this.nodes.leaderboardHelp) {
      this.nodes.leaderboardHelp.title = this.buildLeaderboardHelpText();
      this.nodes.leaderboardHelp.dataset.help = this.buildLeaderboardHelpText();
    }
  }

  buildFeaturePresenceHelpText() {
    const metricKey = String(this.state.featurePresenceMetric || '').trim();
    const metricConfig = FEATURE_PRESENCE_METRICS[metricKey] || FEATURE_PRESENCE_METRICS.presence_rate;
    return `${HELP_TEXT.featurePresence} Current metric: ${metricConfig.label} (${metricConfig.description}).`;
  }

  buildLeaderboardHelpText() {
    const weights = normalizeSignalWeights(this.signalWeights, this.config.signalSettings?.weights);
    const listingBasis = this.state.leaderboardListingBasis === 'page_one'
      ? 'Page-one only (news excluded)'
      : 'All listings (news + page one)';
    return `${HELP_TEXT.leaderboard} Formula: (${formatWeightPercent(weights.newsNegative)} × Negative News) + (${formatWeightPercent(weights.organicNegative)} × Negative Organic SERP) + (${formatWeightPercent(weights.topStoriesNegative)} × Negative Top Stories) + (${formatWeightPercent(weights.aioCitationsNegative)} × Negative AIO Citations) + (${formatWeightPercent(weights.paaNegative)} × Negative PAA) + (${formatWeightPercent(weights.videosNegative)} × Negative Videos) + (${formatWeightPercent(weights.perspectivesNegative)} × Negative Perspectives) − (${formatWeightPercent(weights.serpControl)} × Controlled Organic SERP). Window basis: ${listingBasis}.`;
  }

  readWeightsFromControls() {
    return normalizeSignalWeights({
      newsNegative: this.nodes.weightNews?.value,
      organicNegative: this.nodes.weightSerp?.value,
      topStoriesNegative: this.nodes.weightTopStories?.value,
      aioCitationsNegative: this.nodes.weightAio?.value,
      paaNegative: this.nodes.weightPaa?.value,
      videosNegative: this.nodes.weightVideos?.value,
      perspectivesNegative: this.nodes.weightPerspectives?.value,
      serpControl: this.nodes.weightControl?.value,
    }, this.config.signalSettings?.weights);
  }

  syncCalibrationControls() {
    const weights = normalizeSignalWeights(this.signalWeights, this.config.signalSettings?.weights);
    const assign = (input, output, value) => {
      if (input) input.value = String(value);
      if (output) output.textContent = formatWeightPercent(value);
    };
    assign(this.nodes.weightNews, this.nodes.weightNewsValue, weights.newsNegative);
    assign(this.nodes.weightSerp, this.nodes.weightSerpValue, weights.organicNegative);
    assign(this.nodes.weightTopStories, this.nodes.weightTopStoriesValue, weights.topStoriesNegative);
    assign(this.nodes.weightAio, this.nodes.weightAioValue, weights.aioCitationsNegative);
    assign(this.nodes.weightPaa, this.nodes.weightPaaValue, weights.paaNegative);
    assign(this.nodes.weightVideos, this.nodes.weightVideosValue, weights.videosNegative);
    assign(this.nodes.weightPerspectives, this.nodes.weightPerspectivesValue, weights.perspectivesNegative);
    assign(this.nodes.weightControl, this.nodes.weightControlValue, weights.serpControl);
    this.updateHelpText(this.state.selectedEntity);
    this.updateCalibrationMetric();
  }

  setCalibrationOpen(nextOpen) {
    this.state.calibrationOpen = !!nextOpen;
    if (this.nodes.calibrationPanel) this.nodes.calibrationPanel.hidden = !this.state.calibrationOpen;
    if (this.nodes.calibrateToggle) {
      if (this.state.calibrationOpen) this.nodes.calibrateToggle.dataset.variant = 'active';
      else delete this.nodes.calibrateToggle.dataset.variant;
      this.nodes.calibrateToggle.setAttribute('aria-pressed', this.state.calibrationOpen ? 'true' : 'false');
    }
  }

  updateCalibrationMetric() {
    if (!this.nodes.calibrationMetric) return;
    const validRows = this.rows.filter((row) => Number.isFinite(Number(row.stock)));
    if (validRows.length < 5) {
      this.nodes.calibrationMetric.textContent = 'Need at least 5 rows with stock data to calibrate.';
      return;
    }
    const x = [];
    const y = [];
    validRows.forEach((row) => {
      const score = computeCompositeSignal(row, this.signalWeights);
      const downside = -Number(row.stock);
      if (!Number.isFinite(score) || !Number.isFinite(downside)) return;
      x.push(score);
      y.push(downside);
    });
    const corr = pearsonCorrelation(x, y);
    if (corr == null) {
      this.nodes.calibrationMetric.textContent = 'Calibration metric unavailable for this slice.';
      return;
    }
    const direction = corr >= 0 ? 'aligned' : 'inverted';
    this.nodes.calibrationMetric.textContent = `Downside-stock correlation: r=${corr.toFixed(3)} (${x.length} rows, ${direction}).`;
  }

  async applySignalWeights(nextWeights, { message = '' } = {}) {
    this.signalWeights = normalizeSignalWeights(nextWeights, this.config.signalSettings?.weights);
    this.syncCalibrationControls();
    this.rows = this.sortRows(this.rows.map((row) => ({
      ...row,
      riskScore: computeCompositeSignal(row, this.signalWeights),
    })));
    this.renderSummary();
    this.renderSelected();
    this.renderTable();
    await this.updateCharts();
    if (message) this.setFeedback(message);
  }

  applyModalState(nextOpen) {
    if (!this.nodes.entityModal) return;
    this.nodes.entityModal.hidden = !nextOpen;
  }

  closeEntityModal() {
    this.modalLoadToken += 1;
    this.modalEntity = null;
    this.modalState = null;
    this.applyModalState(false);
  }

  modalEntityParam() {
    return this.config.entityType === 'ceo' ? 'ceo' : 'brand';
  }

  sentimentBadgeClass(value) {
    const sentiment = String(value || '').toLowerCase();
    if (sentiment.includes('positive')) return 'positive';
    if (sentiment.includes('negative')) return 'negative';
    if (sentiment.includes('controlled')) return 'controlled';
    if (sentiment.includes('uncontrolled')) return 'uncontrolled';
    return 'neutral';
  }

  normalizeSentimentValue(value) {
    const sentiment = String(value || '').trim().toLowerCase();
    if (sentiment === 'positive' || sentiment === 'neutral' || sentiment === 'negative') return sentiment;
    return 'neutral';
  }

  normalizeControlValue(value) {
    const control = String(value || '').trim().toLowerCase();
    if (control === 'controlled' || control === 'true' || control === '1') return 'controlled';
    if (control === 'uncontrolled' || control === 'false' || control === '0') return 'uncontrolled';
    return '';
  }

  buildModalEditFlags({ sentimentOverride = '', controlOverride = '', llmLabel = '' } = {}) {
    const parts = [];
    if (sentimentOverride || controlOverride) parts.push('<span title="Manually edited">Manually edited</span>');
    if (llmLabel) parts.push('<span title="AI enriched">AI enriched</span>');
    if (!parts.length) return '';
    return `<div class="entity-edit-flags">${parts.join('')}</div>`;
  }

  updateModalEditClasses(card) {
    if (!card) return;
    const riskSel = card.querySelector('.entity-edit-risk');
    if (riskSel) {
      riskSel.classList.remove('sentiment-positive', 'sentiment-neutral', 'sentiment-negative');
      const sentiment = this.normalizeSentimentValue(riskSel.value);
      riskSel.classList.add(`sentiment-${sentiment}`);
    }
    const ctrlSel = card.querySelector('.entity-edit-controlled');
    if (ctrlSel) {
      ctrlSel.classList.remove('controlled', 'uncontrolled');
      const control = this.normalizeControlValue(ctrlSel.value);
      if (control) ctrlSel.classList.add(control);
    }
  }

  setModalCardFlags(card) {
    if (!card) return;
    const sentimentOverride = String(card.getAttribute('data-sentiment-override') || '').trim();
    const controlOverride = String(card.getAttribute('data-control-override') || '').trim();
    const llmLabel = String(card.getAttribute('data-llm-label') || '').trim();
    const flagsHtml = this.buildModalEditFlags({ sentimentOverride, controlOverride, llmLabel });
    let flagsEl = card.querySelector('.entity-edit-flags');
    if (flagsHtml) {
      if (!flagsEl) {
        flagsEl = document.createElement('div');
        flagsEl.className = 'entity-edit-flags';
        const titleEl = card.querySelector('.entity-url-title');
        if (titleEl && titleEl.parentNode) {
          titleEl.parentNode.insertBefore(flagsEl, titleEl.nextSibling);
        } else {
          card.appendChild(flagsEl);
        }
      }
      flagsEl.innerHTML = flagsHtml.replace('<div class="entity-edit-flags">', '').replace('</div>', '');
    } else if (flagsEl) {
      flagsEl.remove();
    }
  }

  async postOverride(payload) {
    const response = await fetch('/api/internal/overrides', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      let detail = `HTTP ${response.status}`;
      try {
        const data = await response.json();
        detail = data?.error || detail;
      } catch (_error) {
        try {
          detail = await response.text();
        } catch (_error2) {
          // Ignore parsing failures.
        }
      }
      throw new Error(detail || 'Failed to save override');
    }
    return response.json();
  }

  applyModalRowOverride(card, {
    sentimentOverride,
    controlOverride,
    sentiment,
    control,
  }) {
    if (!this.modalState) return;
    const rowIndex = Number(card.getAttribute('data-modal-row-index'));
    if (!Number.isInteger(rowIndex) || rowIndex < 0 || rowIndex >= this.modalState.rows.length) return;
    const row = this.modalState.rows[rowIndex];
    if (!row || typeof row !== 'object') return;
    if (sentimentOverride !== undefined) {
      const normalizedOverride = String(sentimentOverride || '').trim();
      row.sentiment_override = normalizedOverride;
      const effectiveSentiment = normalizedOverride || this.normalizeSentimentValue(sentiment || row.sentiment);
      if (effectiveSentiment) row.sentiment = effectiveSentiment;
    }
    if (controlOverride !== undefined) {
      const normalizedOverride = String(controlOverride || '').trim();
      row.control_override = normalizedOverride;
      const normalizedControl = this.normalizeControlValue(control || row.controlled || row.control_class);
      const effectiveControl = normalizedOverride || normalizedControl || 'uncontrolled';
      row.controlled = effectiveControl;
      row.control_class = effectiveControl;
    }
  }

  buildUrlCard({ title, url, meta = '', snippet = '', badges = [], edit = null, rowIndex = -1 }) {
    const safeTitle = escapeHtml(title || '(untitled)');
    const safeUrl = String(url || '').trim();
    const safeMeta = escapeHtml(meta || '');
    const safeSnippet = escapeHtml(snippet || '');
    const badgeHtml = badges
      .filter(Boolean)
      .map((badge) => `<span class="badge ${this.sentimentBadgeClass(badge)}">${escapeHtml(badge)}</span>`)
      .join('');
    const canEdit = !!(this.config.isInternal && edit && (edit.mentionId || edit.serpResultId || edit.serpFeatureItemId));
    const sentiment = this.normalizeSentimentValue(edit?.sentiment);
    const control = this.normalizeControlValue(edit?.control) || 'uncontrolled';
    const sentimentOverride = this.normalizeSentimentValue(edit?.sentimentOverride || sentiment);
    const controlOverride = this.normalizeControlValue(edit?.controlOverride || control);
    const flagsHtml = this.buildModalEditFlags({
      sentimentOverride: String(edit?.sentimentOverride || '').trim(),
      controlOverride: String(edit?.controlOverride || '').trim(),
      llmLabel: String(edit?.llmLabel || '').trim(),
    });
    const controlsHtml = canEdit
      ? `
        <div class="entity-edit-controls">
          <select class="entity-edit-risk sentiment-${escapeHtml(sentimentOverride)}">
            <option value="positive" ${sentimentOverride === 'positive' ? 'selected' : ''}>Positive</option>
            <option value="neutral" ${sentimentOverride === 'neutral' ? 'selected' : ''}>Neutral</option>
            <option value="negative" ${sentimentOverride === 'negative' ? 'selected' : ''}>Negative</option>
          </select>
          <select class="entity-edit-controlled ${escapeHtml(controlOverride)}">
            <option value="controlled" ${controlOverride === 'controlled' ? 'selected' : ''}>Controlled</option>
            <option value="uncontrolled" ${controlOverride === 'uncontrolled' ? 'selected' : ''}>Uncontrolled</option>
          </select>
          <span class="entity-edit-status muted"></span>
        </div>
      `
      : (badgeHtml ? `<div class="entity-url-badges">${badgeHtml}</div>` : '');
    const mentionType = String(edit?.mentionType || '').trim();
    const mentionId = String(edit?.mentionId || '').trim();
    const serpResultId = String(edit?.serpResultId || '').trim();
    const serpFeatureItemId = String(edit?.serpFeatureItemId || '').trim();
    return `
      <article
        class="entity-url-card"
        data-modal-row-index="${rowIndex}"
        data-mention-type="${escapeHtml(mentionType)}"
        data-mention-id="${escapeHtml(mentionId)}"
        data-serp-result-id="${escapeHtml(serpResultId)}"
        data-serp-feature-item-id="${escapeHtml(serpFeatureItemId)}"
        data-current-sentiment="${escapeHtml(sentiment)}"
        data-current-control="${escapeHtml(control)}"
        data-sentiment-override="${escapeHtml(String(edit?.sentimentOverride || '').trim())}"
        data-control-override="${escapeHtml(String(edit?.controlOverride || '').trim())}"
        data-llm-label="${escapeHtml(String(edit?.llmLabel || '').trim())}"
      >
        ${safeMeta ? `<p class="entity-url-meta">${safeMeta}</p>` : ''}
        <h4 class="entity-url-title">${
          safeUrl
            ? `<a href="${escapeHtml(safeUrl)}" target="_blank" rel="noopener">${safeTitle}</a>`
            : safeTitle
        }</h4>
        ${safeSnippet ? `<p class="entity-url-snippet">${safeSnippet}</p>` : ''}
        ${flagsHtml}
        ${controlsHtml}
      </article>
    `;
  }

  renderModalBody() {
    if (!this.nodes.entityModalBody || !this.modalState) return;
    const { row, date, tab, rows, total, loading, done } = this.modalState;
    const selectedFeatureType = normalizeFeatureType(this.modalState.featureType);
    const isOrganicFeatureView = tab === 'features' && selectedFeatureType === 'organic';
    const selectedFeatureLabel = selectedFeatureType ? featureTypeLabel(selectedFeatureType) : 'All features';
    const contextLabel = tab === 'features'
      ? `Date: ${date} · Feature filter: ${selectedFeatureLabel}`
      : `Date: ${date}`;
    const tabButton = (key, label) => `<button type="button" class="entity-modal-tab${tab === key ? ' active' : ''}" data-modal-tab="${key}">${label}</button>`;
    const featureFilterHtml = tab === 'features'
      ? `
        <div class="entity-modal-feature-filters">
          ${[
            { key: '', label: 'All features' },
            ...FEATURE_MODAL_FILTER_ORDER.map((key) => ({ key, label: featureTypeLabel(key) })),
          ].map((option) => `
            <button
              type="button"
              class="entity-modal-filter${(selectedFeatureType || '') === option.key ? ' active' : ''}"
              data-modal-feature-filter="${escapeHtml(option.key || '__all__')}"
            >${escapeHtml(option.label)}</button>
          `).join('')}
        </div>
      `
      : '';
    const cards = rows.length
      ? rows.map((item, rowIndex) => {
        const isOrganicRow = String(item?.modal_kind || '').trim() === 'organic_serp';
        if (tab === 'headlines') {
          const meta = [item.source, item.published_date].filter(Boolean).join(' · ');
          return this.buildUrlCard({
            title: item.title,
            url: item.url,
            meta,
            badges: [item.sentiment, item.control_class],
            rowIndex,
            edit: {
              mentionType: this.config.entityType === 'ceo' ? 'ceo_article' : 'company_article',
              mentionId: item.mention_id,
              sentiment: item.sentiment,
              control: item.control_class,
              sentimentOverride: item.sentiment_override,
              controlOverride: item.control_override,
              llmLabel: item.llm_label,
            },
          });
        }
        if (isOrganicFeatureView || isOrganicRow) {
          const meta = [`Position ${item.position || 'n/a'}`, item.published_date].filter(Boolean).join(' · ');
          return this.buildUrlCard({
            title: item.title,
            url: item.url,
            meta,
            snippet: item.snippet,
            badges: [item.sentiment, item.controlled],
            rowIndex,
            edit: {
              mentionType: 'serp_result',
              serpResultId: item.serp_result_id,
              sentiment: item.sentiment,
              control: item.controlled,
              sentimentOverride: item.sentiment_override,
              controlOverride: item.control_override,
              llmLabel: item.llm_label,
            },
          });
        }
        const meta = [featureTypeLabel(item.feature_type), item.domain, item.published_date].filter(Boolean).join(' · ');
        return this.buildUrlCard({
          title: item.title,
          url: item.url,
          meta,
          snippet: item.snippet,
          badges: [item.sentiment, item.control_class],
          rowIndex,
          edit: {
            mentionType: 'serp_feature_item',
            serpFeatureItemId: item.id,
            sentiment: item.sentiment,
            control: item.control_class,
            sentimentOverride: item.sentiment_override,
            controlOverride: item.control_override,
            llmLabel: item.llm_label,
          },
        });
      }).join('')
      : (loading ? '<p class="muted">Loading URLs…</p>' : '<p class="muted">No URLs found for this tab/date.</p>');
    const showMore = !done && rows.length > 0;
    const totalLabel = Number.isFinite(total) ? `Showing ${rows.length} of ${total}` : `Showing ${rows.length}`;
    this.nodes.entityModalBody.innerHTML = `
      <p class="entity-copy" style="margin:0 0 10px;max-width:none;">${escapeHtml(contextLabel)}</p>
      <div class="entity-summary-grid">
        <article class="entity-stat">
          <p class="entity-stat-label">Negative News</p>
          <p class="entity-stat-value">${formatPercent(row.negNews)}</p>
          <p class="entity-stat-subcopy">${formatInteger(row.negNewsCount)} / ${formatInteger(row.newsTotal)} negative articles</p>
        </article>
        <article class="entity-stat">
          <p class="entity-stat-label">Negative Organic SERP</p>
          <p class="entity-stat-value">${formatPercent(row.negSerp)}</p>
          <p class="entity-stat-subcopy">${formatInteger(row.negSerpCount)} / ${formatInteger(row.serpTotal)} negative results</p>
        </article>
        <article class="entity-stat">
          <p class="entity-stat-label">Negative SERP Features</p>
          <p class="entity-stat-value">${formatPercent(row.negSerpFeatures)}</p>
          <p class="entity-stat-subcopy">${formatInteger(row.negSerpFeatureCount)} / ${formatInteger(row.serpFeatureTotal)} negative feature URLs</p>
        </article>
        <article class="entity-stat">
          <p class="entity-stat-label">Composite Signal</p>
          <p class="entity-stat-value">${formatPercent(row.riskScore, 1, 'N/A')}</p>
          <p class="entity-stat-subcopy">${escapeHtml(row.risk || 'N/A')} rating · weighted blend</p>
        </article>
      </div>
      <div class="entity-modal-toolbar">
        <div class="entity-modal-tabs">
          ${tabButton('headlines', 'Headlines')}
          ${tabButton('features', 'Features')}
        </div>
        <span class="entity-modal-count muted">${escapeHtml(totalLabel)}</span>
      </div>
      ${featureFilterHtml}
      <div class="entity-modal-list">${cards}</div>
      <div class="entity-modal-pager">
        <button type="button" class="entity-action" data-modal-more ${showMore && !loading ? '' : 'disabled'}>${loading ? 'Loading…' : 'Load More'}</button>
      </div>
    `;
  }

  async fetchModalRows({ tab, date, entityName, offset, limit, featureType = '' }) {
    const base = `/api/v1`;
    const entityParam = this.modalEntityParam();
    const normalizedFeatureType = normalizeFeatureType(featureType);
    let url = '';
    let pageCount = 0;
    if (tab === 'headlines') {
      url = `${base}/processed_articles?date=${encodeURIComponent(date)}&entity=${encodeURIComponent(entityParam)}&kind=modal&entity_name=${encodeURIComponent(entityName)}&limit=${limit}&offset=${offset}`;
    } else if (tab === 'features' && normalizedFeatureType === 'organic') {
      url = `${base}/processed_serps?date=${encodeURIComponent(date)}&entity=${encodeURIComponent(entityParam)}&kind=modal&entity_name=${encodeURIComponent(entityName)}&limit=${limit}&offset=${offset}`;
      const response = await fetch(url, { credentials: 'same-origin' });
      if (!response.ok) throw new Error(`Request failed (${response.status})`);
      const payload = await response.json();
      const rawRows = Array.isArray(payload?.rows) ? payload.rows : [];
      const rows = normalizeOrganicModalRows(rawRows);
      pageCount = rawRows.length;
      return {
        rows,
        total: Number.isFinite(Number(payload?.total)) ? Number(payload.total) : null,
        pageCount,
      };
    } else if (tab === 'features' && !normalizedFeatureType) {
      const featureUrl = `${base}/serp_feature_items?date=${encodeURIComponent(date)}&entity=${encodeURIComponent(entityParam)}&entity_name=${encodeURIComponent(entityName)}&limit=${limit}&offset=${offset}`;
      const featurePromise = fetch(featureUrl, { credentials: 'same-origin' });
      const includeOrganic = Number(offset) === 0;
      const organicPromise = includeOrganic
        ? fetch(`${base}/processed_serps?date=${encodeURIComponent(date)}&entity=${encodeURIComponent(entityParam)}&kind=modal&entity_name=${encodeURIComponent(entityName)}&limit=200&offset=0`, { credentials: 'same-origin' })
        : null;
      const [featureResponse, organicResponse] = await Promise.all([featurePromise, organicPromise]);
      if (!featureResponse.ok) throw new Error(`Request failed (${featureResponse.status})`);
      if (organicResponse && !organicResponse.ok) throw new Error(`Request failed (${organicResponse.status})`);
      const featurePayload = await featureResponse.json();
      const featureRows = Array.isArray(featurePayload)
        ? featurePayload
        : (Array.isArray(featurePayload?.rows) ? featurePayload.rows : []);
      const organicPayload = organicResponse ? await organicResponse.json() : null;
      const organicRows = organicPayload
        ? normalizeOrganicModalRows(Array.isArray(organicPayload?.rows) ? organicPayload.rows : [])
        : [];
      pageCount = featureRows.length;
      return {
        rows: includeOrganic ? [...organicRows, ...featureRows] : featureRows,
        total: null,
        pageCount,
      };
    } else {
      const featureQuery = normalizedFeatureType ? `&feature_type=${encodeURIComponent(normalizedFeatureType)}` : '';
      url = `${base}/serp_feature_items?date=${encodeURIComponent(date)}&entity=${encodeURIComponent(entityParam)}&entity_name=${encodeURIComponent(entityName)}&limit=${limit}&offset=${offset}${featureQuery}`;
    }
    const response = await fetch(url, { credentials: 'same-origin' });
    if (!response.ok) throw new Error(`Request failed (${response.status})`);
    const payload = await response.json();
    if (tab === 'features') {
      const rows = Array.isArray(payload) ? payload : (Array.isArray(payload?.rows) ? payload.rows : []);
      pageCount = rows.length;
      return {
        rows,
        total: Number.isFinite(Number(payload?.total)) ? Number(payload.total) : null,
        pageCount,
      };
    }
    const rows = Array.isArray(payload?.rows) ? payload.rows : [];
    pageCount = rows.length;
    return {
      rows,
      total: Number.isFinite(Number(payload?.total)) ? Number(payload.total) : null,
      pageCount,
    };
  }

  async loadEntityModalRows({ reset = false } = {}) {
    if (!this.modalState) return;
    const token = ++this.modalLoadToken;
    const selectedFeatureType = normalizeFeatureType(this.modalState.featureType);
    const nextOffset = reset
      ? 0
      : (
        this.modalState.tab === 'features' && !selectedFeatureType
          ? this.modalState.rows.filter((item) => String(item?.modal_kind || '').trim() !== 'organic_serp').length
          : this.modalState.offset
      );
    this.modalState.loading = true;
    this.renderModalBody();
    try {
      const result = await this.fetchModalRows({
        tab: this.modalState.tab,
        date: this.modalState.date,
        entityName: this.modalState.row.entity,
        offset: nextOffset,
        limit: this.modalState.limit,
        featureType: this.modalState.featureType,
      });
      if (!this.modalState || token !== this.modalLoadToken) return;
      const incomingRows = Array.isArray(result.rows) ? result.rows : [];
      this.modalState.rows = reset ? incomingRows : [...this.modalState.rows, ...incomingRows];
      this.modalState.offset = this.modalState.tab === 'features' && !selectedFeatureType
        ? this.modalState.rows.filter((item) => String(item?.modal_kind || '').trim() !== 'organic_serp').length
        : this.modalState.rows.length;
      this.modalState.total = result.total;
      const doneByTotal = Number.isFinite(result.total) && this.modalState.rows.length >= result.total;
      const resultPageCount = Number.isFinite(Number(result.pageCount))
        ? Number(result.pageCount)
        : incomingRows.length;
      const doneByBatch = resultPageCount < this.modalState.limit;
      this.modalState.done = !!(doneByTotal || doneByBatch);
      this.modalState.loading = false;
      this.renderModalBody();
    } catch (error) {
      if (!this.modalState || token !== this.modalLoadToken) return;
      this.modalState.loading = false;
      this.modalState.done = true;
      this.modalState.rows = reset ? [] : this.modalState.rows;
      this.renderModalBody();
      this.setFeedback(error?.message || 'Failed to load URLs for modal.', 'error');
    }
  }

  async setModalTab(tabKey) {
    if (!this.modalState) return;
    if (!['headlines', 'features'].includes(tabKey)) return;
    if (this.modalState.tab === tabKey) return;
    this.modalState.tab = tabKey;
    this.modalState.rows = [];
    this.modalState.offset = 0;
    this.modalState.total = null;
    this.modalState.done = false;
    await this.loadEntityModalRows({ reset: true });
  }

  async setModalFeatureType(featureType) {
    if (!this.modalState) return;
    const normalized = normalizeFeatureType(featureType);
    const nextFeatureType = normalized || '';
    if (this.modalState.tab !== 'features') {
      this.modalState.featureType = nextFeatureType;
      this.modalState.featureDisplayName = nextFeatureType ? featureTypeLabel(nextFeatureType) : 'All features';
      await this.setModalTab('features');
      return;
    }
    if ((this.modalState.featureType || '') === nextFeatureType) return;
    this.modalState.featureType = nextFeatureType;
    this.modalState.featureDisplayName = nextFeatureType ? featureTypeLabel(nextFeatureType) : 'All features';
    this.modalState.rows = [];
    this.modalState.offset = 0;
    this.modalState.total = null;
    this.modalState.done = false;
    await this.loadEntityModalRows({ reset: true });
  }

  openEntityModal(row, context = {}) {
    if (!row || !this.nodes.entityModal) return;
    this.modalEntity = row.entity;
    const legacyUrl = this.buildLegacyUrl(row);
    const dashboardHref = `${legacyUrl.pathname}${legacyUrl.search}${legacyUrl.hash}`;
    let normalizedFeatureType = normalizeFeatureType(context.featureType || context.feature || context.rawFeature);
    const requestedTab = String(context.tab || '').trim().toLowerCase();
    const normalizedTab = requestedTab === 'serp' ? 'features' : requestedTab;
    if (requestedTab === 'serp' && !normalizedFeatureType) normalizedFeatureType = 'organic';
    const featureDisplayName = String(
      context.featureDisplayName
      || context.feature
      || (normalizedFeatureType ? featureTypeLabel(normalizedFeatureType) : ''),
    ).trim();
    this.modalState = {
      row,
      date: String(context.date || this.state.date || '').trim(),
      tab: normalizedTab || (normalizedFeatureType || featureDisplayName ? 'features' : 'headlines'),
      featureType: normalizedFeatureType,
      featureDisplayName,
      rows: [],
      offset: 0,
      limit: 25,
      total: null,
      loading: false,
      done: false,
    };
    if (this.nodes.entityModalTitle) this.nodes.entityModalTitle.textContent = `${row.entity} Snapshot`;
    if (this.nodes.entityModalOpenLink) this.nodes.entityModalOpenLink.href = dashboardHref;
    this.applyModalState(true);
    this.renderModalBody();
    this.loadEntityModalRows({ reset: true });
  }

  async autoFitSignalWeights() {
    const validRows = this.rows.filter((row) => Number.isFinite(Number(row.stock)));
    if (validRows.length < 8) {
      this.setFeedback('Auto-fit needs at least 8 rows with stock data.', 'error');
      return;
    }
    const calibrationCfg = this.config.signalSettings?.calibration || {};
    const iterations = Math.max(200, Number(calibrationCfg.autoIterations) || 1200);
    const controlMax = clamp(Number(calibrationCfg.controlWeightMax) || 0.5, 0, 1);
    const scoreWeights = (weights) => {
      const x = [];
      const y = [];
      validRows.forEach((row) => {
        const score = computeCompositeSignal(row, weights);
        const downside = -Number(row.stock);
        if (!Number.isFinite(score) || !Number.isFinite(downside)) return;
        x.push(score);
        y.push(downside);
      });
      return pearsonCorrelation(x, y);
    };

    let bestWeights = normalizeSignalWeights(this.signalWeights, this.config.signalSettings?.weights);
    let bestCorr = scoreWeights(bestWeights);
    if (!Number.isFinite(bestCorr)) bestCorr = -Infinity;

    for (let index = 0; index < iterations; index += 1) {
      const newsRaw = Math.random();
      const organicRaw = Math.random();
      const topStoriesRaw = Math.random();
      const aioRaw = Math.random();
      const paaRaw = Math.random();
      const videosRaw = Math.random();
      const perspectivesRaw = Math.random();
      const total = newsRaw + organicRaw + topStoriesRaw + aioRaw + paaRaw + videosRaw + perspectivesRaw || 1;
      const candidate = {
        newsNegative: newsRaw / total,
        organicNegative: organicRaw / total,
        topStoriesNegative: topStoriesRaw / total,
        aioCitationsNegative: aioRaw / total,
        paaNegative: paaRaw / total,
        videosNegative: videosRaw / total,
        perspectivesNegative: perspectivesRaw / total,
        serpControl: Math.random() * controlMax,
      };
      const corr = scoreWeights(candidate);
      if (Number.isFinite(corr) && corr > bestCorr) {
        bestCorr = corr;
        bestWeights = candidate;
      }
    }

    await this.applySignalWeights(bestWeights, {
      message: `Auto-fit complete (r=${Number.isFinite(bestCorr) ? bestCorr.toFixed(3) : 'n/a'}).`,
    });
  }

  on(target, type, listener, options) {
    target.addEventListener(type, listener, options);
    this.cleanups.push(() => target.removeEventListener(type, listener, options));
  }

  readUrlState() {
    const url = this.getDirectUrl();
    const params = new URLSearchParams(url.search);
    const requestedDays = Number(params.get('days'));
    if (this.config.lookbackOptions.includes(requestedDays)) {
      this.state.days = requestedDays;
    }
    this.state.date = String(params.get('date') || '').trim();
    this.state.query = String(params.get('q') || params.get(this.config.legacyFilterParam) || '').trim();
    this.state.selectedEntity = String(params.get('entity') || '').trim();
  }

  updateUrl({ replace = true } = {}) {
    const nextUrl = new URL(this.getDirectUrl().href);
    if (this.state.days !== this.config.defaultDays) {
      nextUrl.searchParams.set('days', String(this.state.days));
    } else {
      nextUrl.searchParams.delete('days');
    }

    if (this.state.date) nextUrl.searchParams.set('date', this.state.date);
    else nextUrl.searchParams.delete('date');

    if (this.state.query) nextUrl.searchParams.set('q', this.state.query);
    else nextUrl.searchParams.delete('q');

    const legacyValue = this.state.query || this.state.selectedEntity;
    if (legacyValue) nextUrl.searchParams.set(this.config.legacyFilterParam, legacyValue);
    else nextUrl.searchParams.delete(this.config.legacyFilterParam);

    if (this.state.selectedEntity) nextUrl.searchParams.set('entity', this.state.selectedEntity);
    else nextUrl.searchParams.delete('entity');

    this.onHistoryChange(nextUrl, { replace });
  }

  bind() {
    this.nodes.lookbacks.forEach((button) => {
      this.on(button, 'click', async () => {
        const nextDays = Number(button.dataset.lookback || this.config.defaultDays);
        if (nextDays === this.state.days) return;
        this.state.days = nextDays;
        await this.load();
      });
    });

    this.on(this.nodes.dateSelect, 'change', async () => {
      this.state.date = this.nodes.dateSelect.value;
      await this.refreshRowsAndVisuals();
    });

    this.on(this.nodes.queryInput, 'input', async () => {
      this.state.query = this.nodes.queryInput.value.trim();
      await this.refreshRowsAndVisuals();
    });

    this.on(this.nodes.resetButton, 'click', async () => {
      this.state.query = '';
      this.state.selectedEntity = '';
      this.state.sortKey = 'riskScore';
      this.state.sortDir = 'desc';
      this.nodes.queryInput.value = '';
      await this.refreshRowsAndVisuals();
    });

    if (this.nodes.calibrateToggle) {
      this.on(this.nodes.calibrateToggle, 'click', () => {
        this.setCalibrationOpen(!this.state.calibrationOpen);
      });
    }

    let weightApplyTimer = 0;
    this.cleanups.push(() => {
      if (weightApplyTimer) {
        window.clearTimeout(weightApplyTimer);
        weightApplyTimer = 0;
      }
    });
    const scheduleWeightApply = () => {
      if (weightApplyTimer) window.clearTimeout(weightApplyTimer);
      weightApplyTimer = window.setTimeout(() => {
        weightApplyTimer = 0;
        this.applySignalWeights(this.readWeightsFromControls())
          .catch((error) => this.setFeedback(error?.message || 'Failed to apply weights.', 'error'));
      }, 120);
    };
    const bindWeight = (input, output) => {
      if (!input) return;
      this.on(input, 'input', () => {
        if (output) output.textContent = formatWeightPercent(input.value);
        scheduleWeightApply();
      });
      this.on(input, 'change', async () => {
        if (weightApplyTimer) {
          window.clearTimeout(weightApplyTimer);
          weightApplyTimer = 0;
        }
        await this.applySignalWeights(this.readWeightsFromControls(), { message: 'Composite weights updated.' });
      });
    };
    bindWeight(this.nodes.weightNews, this.nodes.weightNewsValue);
    bindWeight(this.nodes.weightSerp, this.nodes.weightSerpValue);
    bindWeight(this.nodes.weightTopStories, this.nodes.weightTopStoriesValue);
    bindWeight(this.nodes.weightAio, this.nodes.weightAioValue);
    bindWeight(this.nodes.weightPaa, this.nodes.weightPaaValue);
    bindWeight(this.nodes.weightVideos, this.nodes.weightVideosValue);
    bindWeight(this.nodes.weightPerspectives, this.nodes.weightPerspectivesValue);
    bindWeight(this.nodes.weightControl, this.nodes.weightControlValue);

    if (this.nodes.calibrationReset) {
      this.on(this.nodes.calibrationReset, 'click', async () => {
        this.activePresetKey = '__default__';
        this.renderPresetOptions(this.activePresetKey);
        await this.applySignalWeights(this.config.signalSettings?.weights, { message: 'Weights reset to defaults.' });
      });
    }
    if (this.nodes.calibrationAuto) {
      this.on(this.nodes.calibrationAuto, 'click', async () => {
        this.nodes.calibrationAuto.disabled = true;
        try {
          await this.autoFitSignalWeights();
        } finally {
          this.nodes.calibrationAuto.disabled = false;
        }
      });
    }

    if (this.nodes.presetSelect) {
      this.on(this.nodes.presetSelect, 'change', async () => {
        this.activePresetKey = String(this.nodes.presetSelect.value || '__default__');
        const weights = this.activePresetKey === '__default__'
          ? this.config.signalSettings?.weights
          : this.savedPresets[this.activePresetKey];
        await this.applySignalWeights(weights || this.config.signalSettings?.weights, {
          message: this.activePresetKey === '__default__'
            ? 'Loaded default weights.'
            : `Loaded preset: ${this.activePresetKey}.`,
        });
        this.renderPresetOptions(this.activePresetKey);
      });
    }

    if (this.nodes.presetSave) {
      this.on(this.nodes.presetSave, 'click', async () => {
        const typedName = String(this.nodes.presetName?.value || '').trim();
        const isExplicitName = !!typedName;
        let presetName = typedName || this.activePresetKey;
        if (!presetName || presetName === '__default__') {
          presetName = buildAutoPresetName(`${this.config.label} preset`);
        }
        if (!presetName || presetName === '__default__') return;
        if (!isExplicitName && presetName !== this.activePresetKey) {
          const baseName = presetName;
          let suffix = 2;
          while (this.savedPresets[presetName]) {
            presetName = `${baseName} (${suffix})`;
            suffix += 1;
          }
        }
        const normalizedWeights = normalizeSignalWeights(this.signalWeights, this.config.signalSettings?.weights);
        if (this.config.isInternal) {
          try {
            await this.saveSharedPreset(presetName, normalizedWeights);
            this.sharedPresetsEnabled = true;
            await this.loadSharedPresets({ quiet: true });
            this.renderPresetOptions(presetName);
            this.setFeedback(`Saved shared preset: ${presetName}.`);
            return;
          } catch (error) {
            this.sharedPresetsEnabled = false;
            this.savedPresets[presetName] = normalizedWeights;
            this.persistPresets();
            this.renderPresetOptions(presetName);
            this.setFeedback(`Saved locally only (shared save failed: ${error?.message || 'unknown error'}).`, 'error');
            return;
          }
        }
        this.savedPresets[presetName] = normalizedWeights;
        this.persistPresets();
        this.renderPresetOptions(presetName);
        this.setFeedback(`Saved preset: ${presetName}.`);
      });
    }
    if (this.nodes.presetName && this.nodes.presetSave) {
      this.on(this.nodes.presetName, 'keydown', (event) => {
        if (event.key !== 'Enter') return;
        event.preventDefault();
        this.nodes.presetSave.click();
      });
    }

    if (this.nodes.presetDelete) {
      this.on(this.nodes.presetDelete, 'click', async () => {
        const presetName = this.activePresetKey;
        if (!presetName || presetName === '__default__') return;
        if (this.config.isInternal) {
          try {
            await this.deleteSharedPreset(presetName);
            this.sharedPresetsEnabled = true;
            await this.loadSharedPresets({ quiet: true });
            this.activePresetKey = '__default__';
            this.renderPresetOptions(this.activePresetKey);
            await this.applySignalWeights(this.config.signalSettings?.weights, { message: `Deleted shared preset: ${presetName}.` });
            return;
          } catch (error) {
            this.sharedPresetsEnabled = false;
            delete this.savedPresets[presetName];
            this.persistPresets();
            this.activePresetKey = '__default__';
            this.renderPresetOptions(this.activePresetKey);
            await this.applySignalWeights(this.config.signalSettings?.weights, {
              message: `Deleted locally only (shared delete failed: ${error?.message || 'unknown error'}).`,
            });
            return;
          }
        }
        delete this.savedPresets[presetName];
        this.persistPresets();
        this.activePresetKey = '__default__';
        this.renderPresetOptions(this.activePresetKey);
        await this.applySignalWeights(this.config.signalSettings?.weights, { message: `Deleted preset: ${presetName}.` });
      });
    }

    if (this.nodes.refreshButton) {
      this.on(this.nodes.refreshButton, 'click', async () => {
        try {
          this.setFeedback('Refreshing internal aggregates…');
          const status = await runInternalRefresh(this.nodes.feedback);
          clearAllNativeDataStores();
          this.store = getEntityStore(this.config);
          this.supplementalLoadScheduled = false;
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
        this.state.sortDir = sortKey === 'entity' || sortKey === 'company' ? 'asc' : 'desc';
      }
      await this.refreshRowsAndVisuals({ updateCharts: false, replaceUrl: false });
    });

    this.on(this.nodes.tableBody, 'click', async (event) => {
      const detailBtn = event.target.closest('[data-entity-modal-btn]');
      if (detailBtn) {
        event.preventDefault();
        event.stopPropagation();
        const entity = String(detailBtn.getAttribute('data-entity-modal-btn') || '').trim();
        const rowData = this.rows.find((row) => row.entity === entity);
        if (rowData) this.openEntityModal(rowData, { date: this.state.date });
        return;
      }
      const row = event.target.closest('[data-entity-row]');
      if (!row) return;
      const entity = row.getAttribute('data-entity-row') || '';
      this.state.selectedEntity = this.state.selectedEntity === entity ? '' : entity;
      await this.refreshRowsAndVisuals();
    });

    if (this.nodes.entityModalClose) {
      this.on(this.nodes.entityModalClose, 'click', () => this.closeEntityModal());
    }
    if (this.nodes.entityModalBackdrop) {
      this.on(this.nodes.entityModalBackdrop, 'click', () => this.closeEntityModal());
    }
    if (this.nodes.entityModalFocus) {
      this.on(this.nodes.entityModalFocus, 'click', async () => {
        if (!this.modalEntity) return;
        this.state.selectedEntity = this.modalEntity;
        this.closeEntityModal();
        await this.refreshRowsAndVisuals();
      });
    }
    if (this.nodes.entityModalBody) {
      this.on(this.nodes.entityModalBody, 'click', async (event) => {
        const tabButton = event.target.closest('[data-modal-tab]');
        if (tabButton) {
          event.preventDefault();
          const tab = String(tabButton.getAttribute('data-modal-tab') || '').trim();
          await this.setModalTab(tab);
          return;
        }
        const moreButton = event.target.closest('[data-modal-more]');
        if (moreButton) {
          event.preventDefault();
          if (moreButton.hasAttribute('disabled')) return;
          await this.loadEntityModalRows({ reset: false });
          return;
        }
        const featureFilterButton = event.target.closest('[data-modal-feature-filter]');
        if (featureFilterButton) {
          event.preventDefault();
          const rawValue = String(featureFilterButton.getAttribute('data-modal-feature-filter') || '').trim();
          const featureType = rawValue === '__all__' ? '' : rawValue;
          await this.setModalFeatureType(featureType);
        }
      });
      this.on(this.nodes.entityModalBody, 'change', async (event) => {
        const control = event.target.closest('.entity-edit-risk, .entity-edit-controlled');
        if (!control) return;
        const card = control.closest('.entity-url-card');
        if (!card || !this.config.isInternal) return;
        const sentimentSelect = card.querySelector('.entity-edit-risk');
        const controlSelect = card.querySelector('.entity-edit-controlled');
        const statusEl = card.querySelector('.entity-edit-status');
        if (!(sentimentSelect instanceof HTMLSelectElement) || !(controlSelect instanceof HTMLSelectElement)) return;

        const mentionType = String(card.getAttribute('data-mention-type') || '').trim();
        const mentionId = String(card.getAttribute('data-mention-id') || '').trim();
        const serpResultId = String(card.getAttribute('data-serp-result-id') || '').trim();
        const serpFeatureItemId = String(card.getAttribute('data-serp-feature-item-id') || '').trim();
        if (!mentionType) return;

        const previousCurrentSentiment = this.normalizeSentimentValue(card.getAttribute('data-current-sentiment'));
        const previousCurrentControl = this.normalizeControlValue(card.getAttribute('data-current-control')) || 'uncontrolled';
        const previousSentimentOverrideRaw = String(card.getAttribute('data-sentiment-override') || '').trim();
        const previousControlOverrideRaw = String(card.getAttribute('data-control-override') || '').trim();
        const previousSentimentDisplay = this.normalizeSentimentValue(previousSentimentOverrideRaw || previousCurrentSentiment);
        const previousControlDisplay = this.normalizeControlValue(previousControlOverrideRaw || previousCurrentControl) || 'uncontrolled';

        const nextSentiment = this.normalizeSentimentValue(sentimentSelect.value);
        const nextControl = this.normalizeControlValue(controlSelect.value) || 'uncontrolled';
        const noChange = nextSentiment === previousSentimentDisplay && nextControl === previousControlDisplay;
        if (noChange) {
          this.updateModalEditClasses(card);
          return;
        }

        const payload = {
          mention_type: mentionType,
          sentiment_override: nextSentiment,
          control_override: nextControl,
          note: 'dashboard edit',
        };
        if (mentionType === 'serp_result') payload.serp_result_id = serpResultId;
        else if (mentionType === 'serp_feature_item') payload.serp_feature_item_id = serpFeatureItemId;
        else payload.mention_id = mentionId;

        if (
          (mentionType === 'serp_result' && !serpResultId)
          || (mentionType === 'serp_feature_item' && !serpFeatureItemId)
          || ((mentionType === 'company_article' || mentionType === 'ceo_article') && !mentionId)
        ) {
          if (statusEl) {
            statusEl.textContent = 'Missing row id';
            statusEl.classList.remove('ok');
            statusEl.classList.add('error');
          }
          return;
        }

        sentimentSelect.disabled = true;
        controlSelect.disabled = true;
        card.setAttribute('data-sentiment-override', nextSentiment);
        card.setAttribute('data-control-override', nextControl);
        card.setAttribute('data-current-sentiment', nextSentiment);
        card.setAttribute('data-current-control', nextControl);
        this.applyModalRowOverride(card, {
          sentimentOverride: nextSentiment,
          controlOverride: nextControl,
          sentiment: nextSentiment,
          control: nextControl,
        });
        this.updateModalEditClasses(card);
        this.setModalCardFlags(card);
        if (statusEl) {
          statusEl.textContent = 'Saving…';
          statusEl.classList.remove('error', 'ok');
        }

        try {
          await this.postOverride(payload);
          if (statusEl) {
            statusEl.textContent = 'Saved';
            statusEl.classList.remove('error');
            statusEl.classList.add('ok');
          }
          this.setFeedback('Saved override.');
        } catch (error) {
          card.setAttribute('data-sentiment-override', previousSentimentOverrideRaw);
          card.setAttribute('data-control-override', previousControlOverrideRaw);
          card.setAttribute('data-current-sentiment', previousCurrentSentiment);
          card.setAttribute('data-current-control', previousCurrentControl);
          sentimentSelect.value = previousSentimentDisplay;
          controlSelect.value = previousControlDisplay;
          this.applyModalRowOverride(card, {
            sentimentOverride: previousSentimentOverrideRaw,
            controlOverride: previousControlOverrideRaw,
            sentiment: previousCurrentSentiment,
            control: previousCurrentControl,
          });
          this.updateModalEditClasses(card);
          this.setModalCardFlags(card);
          if (statusEl) {
            statusEl.textContent = 'Failed to save';
            statusEl.classList.remove('ok');
            statusEl.classList.add('error');
          }
          this.setFeedback(error?.message || 'Failed to save override.', 'error');
        } finally {
          sentimentSelect.disabled = false;
          controlSelect.disabled = false;
        }
      });
    }
    if (Array.isArray(this.nodes.featureCompositeViewButtons) && this.nodes.featureCompositeViewButtons.length) {
      this.nodes.featureCompositeViewButtons.forEach((button) => {
        this.on(button, 'click', async () => {
          const nextView = String(button.getAttribute('data-view') || '').trim().toLowerCase();
          if (!['bar', 'area'].includes(nextView)) return;
          if (this.state.featureCompositeView === nextView) return;
          this.state.featureCompositeView = nextView;
          this.persistFeatureCompositeView();
          this.syncFeatureCompositeViewControls();
          await this.updateCharts();
        });
      });
    }
    if (this.nodes.featurePresenceMetric) {
      this.on(this.nodes.featurePresenceMetric, 'change', async () => {
        const nextMetric = String(this.nodes.featurePresenceMetric.value || '').trim();
        if (!(nextMetric in FEATURE_PRESENCE_METRICS)) return;
        if (this.state.featurePresenceMetric === nextMetric) return;
        this.state.featurePresenceMetric = nextMetric;
        this.syncFeaturePresenceMetricControl();
        this.updateHelpText(this.state.selectedEntity);
        await this.updateCharts();
      });
    }
    if (Array.isArray(this.nodes.leaderboardScopeButtons) && this.nodes.leaderboardScopeButtons.length) {
      this.nodes.leaderboardScopeButtons.forEach((button) => {
        this.on(button, 'click', async () => {
          const nextScope = String(button.getAttribute('data-scope') || '').trim().toLowerCase();
          if (!['date', 'window'].includes(nextScope)) return;
          if (this.state.leaderboardScope === nextScope) return;
          this.state.leaderboardScope = nextScope;
          this.syncLeaderboardControls();
          await this.updateCharts();
        });
      });
    }
    if (Array.isArray(this.nodes.leaderboardDirectionButtons) && this.nodes.leaderboardDirectionButtons.length) {
      this.nodes.leaderboardDirectionButtons.forEach((button) => {
        this.on(button, 'click', async () => {
          const nextDirection = String(button.getAttribute('data-direction') || '').trim().toLowerCase();
          if (!['worst', 'best'].includes(nextDirection)) return;
          if (this.state.leaderboardDirection === nextDirection) return;
          this.state.leaderboardDirection = nextDirection;
          this.syncLeaderboardControls();
          if (this.state.leaderboardScope !== 'window') return;
          await this.updateCharts();
        });
      });
    }
    if (this.nodes.leaderboardMinVolume) {
      this.on(this.nodes.leaderboardMinVolume, 'change', async () => {
        const nextMin = Math.max(0, Number(this.nodes.leaderboardMinVolume.value) || 0);
        if (this.state.leaderboardMinDailyListings === nextMin) return;
        this.state.leaderboardMinDailyListings = nextMin;
        this.syncLeaderboardControls();
        if (this.state.leaderboardScope !== 'window') return;
        await this.updateCharts();
      });
    }
    if (this.nodes.leaderboardListingBasis) {
      this.on(this.nodes.leaderboardListingBasis, 'change', async () => {
        const nextBasis = String(this.nodes.leaderboardListingBasis.value || '').trim().toLowerCase() === 'page_one'
          ? 'page_one'
          : 'all';
        if (this.state.leaderboardListingBasis === nextBasis) return;
        this.state.leaderboardListingBasis = nextBasis;
        this.syncLeaderboardControls();
        if (this.state.leaderboardScope !== 'window') return;
        await this.updateCharts();
      });
    }
    this.on(document, 'keydown', (event) => {
      if (event.key === 'Escape') this.closeEntityModal();
    });
  }

  setFeedback(message, stateName = '') {
    this.nodes.feedback.textContent = message || '';
    if (stateName) this.nodes.feedback.dataset.state = stateName;
    else delete this.nodes.feedback.dataset.state;
  }

  updateLookbackButtons() {
    this.nodes.lookbacks.forEach((button) => {
      const isActive = Number(button.dataset.lookback) === this.state.days;
      button.setAttribute('aria-pressed', isActive ? 'true' : 'false');
    });
  }

  updateDateSelect() {
    const datesDesc = this.core?.dates ? [...this.core.dates].sort((left, right) => right.localeCompare(left)) : [];
    const options = datesDesc.map((date) => `
      <option value="${escapeHtml(date)}"${date === this.state.date ? ' selected' : ''}>${escapeHtml(date)}</option>
    `).join('');
    this.nodes.dateSelect.innerHTML = options;
    this.nodes.dateSelect.value = this.state.date;
  }

  sortRows(rows) {
    const direction = this.state.sortDir === 'asc' ? 1 : -1;
    const accessor = (row) => {
      switch (this.state.sortKey) {
        case 'entity':
          return row.entity.toLowerCase();
        case 'company':
          return row.company.toLowerCase();
        case 'negNews':
          return row.negNews ?? -1;
        case 'topStories':
          return row.topStories ?? -1;
        case 'negSerp':
          return row.negSerp ?? -1;
        case 'negTopStories':
          return row.negTopStories ?? row.topStories ?? -1;
        case 'negAio':
          return row.negAio ?? -1;
        case 'negPaa':
          return row.negPaa ?? -1;
        case 'negVideos':
          return row.negVideos ?? -1;
        case 'negPerspectives':
          return row.negPerspectives ?? -1;
        case 'negFeatureAll':
          return row.negFeatureAll ?? -1;
        case 'control':
          return row.control ?? -1;
        case 'stock':
          return row.stock ?? -Infinity;
        case 'riskScore':
        default:
          return row.riskScore ?? -Infinity;
      }
    };

    return rows.slice().sort((left, right) => {
      const leftValue = accessor(left);
      const rightValue = accessor(right);
      if (typeof leftValue === 'string' && typeof rightValue === 'string') {
        return leftValue.localeCompare(rightValue) * direction;
      }
      return ((leftValue > rightValue) - (leftValue < rightValue)) * direction;
    });
  }

  selectedRow() {
    return this.rows.find((row) => row.entity === this.state.selectedEntity) || null;
  }

  buildLegacyUrl(row = null) {
    const url = new URL(this.getDirectUrl().href);
    url.searchParams.delete('q');
    url.searchParams.delete('days');
    url.searchParams.delete('date');
    url.searchParams.delete('entity');
    const filterValue = row?.entity || this.state.query || '';
    if (filterValue) url.searchParams.set(this.config.legacyFilterParam, filterValue);
    else url.searchParams.delete(this.config.legacyFilterParam);
    return url;
  }

  renderSummary() {
    const rows = this.rows;
    const selected = this.selectedRow();
    const focusLabel = selected ? selected.entity : `${rows.length} visible`;
    const stats = [
      {
        label: 'Focus',
        value: escapeHtml(focusLabel),
        subcopy: selected
          ? `${escapeHtml(selected.company || selected.entity)} selected`
          : `Top ${escapeHtml(this.config.label.toLowerCase())} for ${escapeHtml(this.state.date)}`,
      },
      {
        label: 'Avg News',
        value: rows.length ? formatPercent(rows.reduce((sum, row) => sum + (row.negNews ?? 0), 0) / rows.length) : 'N/A',
        subcopy: 'Negative article share',
      },
      {
        label: 'Avg SERP',
        value: rows.length ? formatPercent(rows.reduce((sum, row) => sum + (row.negSerp ?? 0), 0) / rows.length) : 'N/A',
        subcopy: 'Negative organic results',
      },
      {
        label: 'Avg Control',
        value: rows.length ? formatPercent(rows.reduce((sum, row) => sum + (row.control ?? 0), 0) / rows.length) : 'N/A',
        subcopy: 'Controlled SERP coverage',
      },
    ];

    this.nodes.summaryGrid.innerHTML = stats.map((stat) => `
      <article class="entity-stat">
        <p class="entity-stat-label">${escapeHtml(stat.label)}</p>
        <p class="entity-stat-value">${stat.value}</p>
        <p class="entity-stat-subcopy">${stat.subcopy}</p>
      </article>
    `).join('');
  }

  renderSelected() {
    const row = this.selectedRow();
    const legacyUrl = this.buildLegacyUrl(row);
    this.nodes.openLink.href = `${legacyUrl.pathname}${legacyUrl.search}${legacyUrl.hash}`;

    if (!row) {
      this.nodes.selectedSpotlight.dataset.empty = 'true';
      this.nodes.selectedSpotlight.innerHTML = `
        <p class="entity-kicker">Selection</p>
        <h3 style="margin:0;font-size:1.35rem;">Pick a ${escapeHtml(this.config.label.slice(0, -1).toLowerCase())} from the native table</h3>
        <p class="entity-copy">We keep the summary charts alive while you move between tabs. Select a row to focus the feature snapshot and build an “open full dashboard” jump target.</p>
      `;
      return;
    }

    this.nodes.selectedSpotlight.dataset.empty = 'false';
    this.nodes.selectedSpotlight.innerHTML = `
      <div class="entity-selected-header">
        <div>
          <p class="entity-kicker">Selected ${escapeHtml(this.config.label.slice(0, -1))}</p>
          <h3>${escapeHtml(row.entity)}</h3>
          <p class="entity-selected-meta">${escapeHtml(row.company || row.entity)}${row.sector ? ` · ${escapeHtml(row.sector)}` : ''}</p>
        </div>
        <span class="entity-pill" data-tone="${escapeHtml(scoreTone(row.risk))}">${escapeHtml(row.risk)} Signal</span>
      </div>
      <div class="entity-selected-stats">
        <article class="entity-stat">
          <p class="entity-stat-label">Negative News</p>
          <p class="entity-stat-value">${formatPercent(row.negNews)}</p>
          <p class="entity-stat-subcopy">Article sentiment share</p>
        </article>
        <article class="entity-stat">
          <p class="entity-stat-label">Top Stories</p>
          <p class="entity-stat-value">${formatPercent(row.topStories)}</p>
          <p class="entity-stat-subcopy">Negative top-story share</p>
        </article>
        <article class="entity-stat">
          <p class="entity-stat-label">Negative SERP</p>
          <p class="entity-stat-value">${formatPercent(row.negSerp)}</p>
          <p class="entity-stat-subcopy">Organic result pressure</p>
        </article>
        <article class="entity-stat">
          <p class="entity-stat-label">Stock Move</p>
          <p class="entity-stat-value">${escapeHtml(formatSignedPercent(row.stock))}</p>
          <p class="entity-stat-subcopy">${escapeHtml(row.ticker || 'Awaiting latest market file')}</p>
        </article>
      </div>
      <a class="entity-action" href="${escapeHtml(`${legacyUrl.pathname}${legacyUrl.search}${legacyUrl.hash}`)}" target="_blank" rel="noopener">Open Full ${escapeHtml(this.config.label.slice(0, -1))} Dashboard</a>
    `;
  }

  renderTable() {
    const arrow = (key) => this.state.sortKey === key ? (this.state.sortDir === 'asc' ? ' ↑' : ' ↓') : '';
    this.nodes.tableHead.innerHTML = `
      <tr>
        ${this.config.tableColumns.map((column) => `
          <th data-sort-key="${escapeHtml(column.key)}">${escapeHtml(column.label)}${arrow(column.key)}</th>
        `).join('')}
        <th>Details</th>
      </tr>
    `;

    if (!this.rows.length) {
      this.nodes.tableBody.innerHTML = `
        <tr>
          <td colspan="${this.config.tableColumns.length + 1}" class="entity-table-empty">No ${escapeHtml(this.config.label.toLowerCase())} match the current filters.</td>
        </tr>
      `;
      this.nodes.tablePill.textContent = '0 rows';
      this.nodes.tableCaption.textContent = 'Try resetting the search or choosing a different date.';
      return;
    }

    const maxScore = Math.max(...this.rows.map((row) => row.riskScore), 0.01);
    this.nodes.tableBody.innerHTML = this.rows.map((row) => `
      <tr data-entity-row="${escapeHtml(row.entity)}" data-selected="${row.entity === this.state.selectedEntity ? 'true' : 'false'}">
        ${this.config.tableColumns.map((column) => {
          switch (column.key) {
            case 'entity':
              return `
                <td>
                  <div class="entity-name-cell">
                    <span class="entity-name-primary">${escapeHtml(row.entity)}</span>
                    ${row.favorite ? '<span class="entity-name-secondary">Favorite</span>' : ''}
                  </div>
                </td>
              `;
            case 'company':
              return `<td>${escapeHtml(row.company || 'N/A')}</td>`;
            case 'negNews':
              return `<td>${escapeHtml(formatPercent(row.negNews))}</td>`;
            case 'topStories':
              return `<td>${escapeHtml(formatPercent(row.topStories))}</td>`;
            case 'negSerp':
              return `<td>${escapeHtml(formatPercent(row.negSerp))}</td>`;
            case 'negTopStories':
              return `<td>${escapeHtml(formatPercent(row.negTopStories ?? row.topStories))}</td>`;
            case 'negAio':
              return `<td>${escapeHtml(formatPercent(row.negAio))}</td>`;
            case 'negPaa':
              return `<td>${escapeHtml(formatPercent(row.negPaa))}</td>`;
            case 'negVideos':
              return `<td>${escapeHtml(formatPercent(row.negVideos))}</td>`;
            case 'negPerspectives':
              return `<td>${escapeHtml(formatPercent(row.negPerspectives))}</td>`;
            case 'negFeatureAll':
              return `<td>${escapeHtml(formatPercent(row.negFeatureAll))}</td>`;
            case 'control':
              return `<td>${escapeHtml(formatPercent(row.control))}</td>`;
            case 'stock': {
              const tone = row.stock > 0 ? 'positive' : row.stock < 0 ? 'negative' : '';
              return `<td><span class="entity-number" data-tone="${tone}">${escapeHtml(formatSignedPercent(row.stock))}</span></td>`;
            }
            case 'riskScore':
            default: {
              const width = Math.max(6, Math.round((row.riskScore / maxScore) * 100));
              return `
                <td>
                  <div class="entity-score-bar">
                    <span class="entity-score-fill" style="width:${width}%"></span>
                  </div>
                </td>
              `;
            }
          }
        }).join('')}
        <td><button type="button" class="entity-action entity-row-detail" data-entity-modal-btn="${escapeHtml(row.entity)}">View</button></td>
      </tr>
    `).join('');

    this.nodes.tablePill.textContent = `${this.rows.length} rows`;
    this.nodes.tableCaption.textContent = this.selectedRow()
      ? `${this.selectedRow().entity} is driving the feature snapshot and line charts.`
      : 'Select a row to focus the charts and prepare a full-dashboard jump link.';
  }

  async updateCharts() {
    const visibleEntities = this.rows.map((row) => row.entity);
    const selectedEntity = this.state.selectedEntity;
    this.updateHelpText(selectedEntity);
    const singularLabel = this.config.label.endsWith('s')
      ? this.config.label.slice(0, -1).toLowerCase()
      : this.config.label.toLowerCase();
    const openDateModalFromChart = (clickedDate, context = {}) => {
      const targetDate = String(clickedDate || '').trim();
      if (!targetDate) return;
      const selectedRow = this.selectedRow();
      if (!selectedRow) {
        this.setFeedback(`Select a ${singularLabel} first, then click a date to open details.`, 'error');
        return;
      }
      this.openEntityModal(selectedRow, { date: targetDate, ...context });
    };

    const leaderboardPromise = this.state.leaderboardScope === 'window'
      ? this.store.buildWindowLeaderboard({
        days: this.state.days,
        visibleEntities,
        signalWeights: this.signalWeights,
        direction: this.state.leaderboardDirection,
        limit: 10,
        restrictToVisible: true,
        minAvgDailyListings: this.state.leaderboardMinDailyListings,
        listingBasis: this.state.leaderboardListingBasis,
      })
      : this.store.buildLeaderboard(this.state.days, this.state.date, this.rows);

    const [newsSeries, serpSeries, featureSnapshot, featureCompositeSeries, featurePresenceSeries, leaderboard] = await Promise.all([
      this.store.buildNewsSeries(this.state.days, visibleEntities, selectedEntity),
      this.store.buildSerpSeries(this.state.days, visibleEntities, selectedEntity),
      this.store.buildFeatureSnapshot(this.state.days, this.state.date, selectedEntity, visibleEntities),
      this.store.buildFeatureCompositeSeries(this.state.days, selectedEntity),
      this.store.buildFeaturePresenceSeries(
        this.state.days,
        visibleEntities,
        selectedEntity,
        this.state.featurePresenceMetric,
      ),
      leaderboardPromise,
    ]);

    const Chart = await ensureChartJs();
    const lineCommon = {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: { intersect: false, mode: 'index' },
      scales: {
        y: {
          beginAtZero: true,
          min: 0,
          max: 100,
          ticks: {
            stepSize: 20,
            callback: (value) => `${value}%`,
            color: '#a6d7dd',
          },
          grid: { color: 'rgba(162,235,243,.08)' },
        },
        x: {
          ticks: { color: '#a6d7dd', maxRotation: 0, autoSkip: true, maxTicksLimit: 8 },
          grid: { color: 'rgba(162,235,243,.05)' },
        },
      },
      plugins: {
        tooltip: {
          callbacks: {
            label: (context) => `${context.dataset.label}: ${formatPercent(context.parsed.y, 1, 'N/A')}`,
          },
        },
      },
    };

    this.upsertChart('news', this.nodes.newsChart, {
      type: 'line',
      data: {
        labels: this.core.dates,
        datasets: [
          {
            label: selectedEntity ? `${selectedEntity} negative news` : 'Visible negative news',
            data: newsSeries,
            borderColor: '#58dbed',
            backgroundColor: 'rgba(88,219,237,.18)',
            tension: 0.28,
            spanGaps: true,
            fill: true,
          },
        ],
      },
      options: {
        ...lineCommon,
        plugins: {
          ...lineCommon.plugins,
          legend: this.buildLegendTriStateOptions('news', { labels: { color: '#eaf5f5' } }),
        },
        onClick: (event, _elements, chart) => {
          const points = chart.getElementsAtEventForMode(event, 'nearest', { intersect: false }, false);
          if (!points || !points.length) return;
          const point = points[0];
          const clickedDate = this.core?.dates?.[point.index] || chart?.data?.labels?.[point.index];
          openDateModalFromChart(clickedDate, { tab: 'headlines' });
        },
      },
    }, Chart);
    this.applyLegendTriStateStyles('news', this.charts.news, { update: false });

    this.upsertChart('serp', this.nodes.serpChart, {
      type: 'line',
      data: {
        labels: serpSeries.dates,
        datasets: [
          {
            label: 'Negative',
            data: serpSeries.negative,
            borderColor: '#ff8261',
            backgroundColor: 'rgba(255,130,97,.18)',
            tension: 0.28,
            spanGaps: true,
          },
          {
            label: 'Controlled',
            data: serpSeries.control,
            borderColor: '#82c616',
            backgroundColor: 'rgba(130,198,22,.18)',
            tension: 0.28,
            spanGaps: true,
          },
        ],
      },
      options: {
        ...lineCommon,
        plugins: {
          ...lineCommon.plugins,
          legend: this.buildLegendTriStateOptions('serp', { labels: { color: '#eaf5f5' } }),
        },
        onClick: (event, _elements, chart) => {
          const points = chart.getElementsAtEventForMode(event, 'nearest', { intersect: false }, false);
          if (!points || !points.length) return;
          const point = points[0];
          const clickedDate = serpSeries?.dates?.[point.index] || chart?.data?.labels?.[point.index];
          openDateModalFromChart(clickedDate, {
            tab: 'features',
            featureType: 'organic',
            featureDisplayName: 'Organic',
          });
        },
      },
    }, Chart);
    this.applyLegendTriStateStyles('serp', this.charts.serp, { update: false });

    const isIndexSnapshot = !selectedEntity;
    const featureDatasets = isIndexSnapshot
      ? [
        {
          label: 'Negative brands',
          data: featureSnapshot.map((row) => row.negativePct),
          borderRadius: 10,
          stack: 'brand-share',
          backgroundColor: featureSnapshot.map(() => 'rgba(255,130,97,.92)'),
        },
        {
          label: 'Positive/neutral brands with feature',
          data: featureSnapshot.map((row) => row.nonNegativePct),
          borderRadius: 10,
          stack: 'brand-share',
          backgroundColor: featureSnapshot.map(() => 'rgba(90,208,225,.72)'),
        },
      ]
      : [
        {
          label: 'Negative feature share',
          data: featureSnapshot.map((row) => row.negativePct),
          borderRadius: 10,
          backgroundColor: featureSnapshot.map(() => 'rgba(255,130,97,.92)'),
        },
      ];

    this.upsertChart('feature', this.nodes.featureChart, {
      type: 'bar',
      data: {
        labels: featureSnapshot.map((row) => row.feature),
        datasets: featureDatasets,
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        indexAxis: 'y',
        onClick: (event, _elements, chart) => {
          if (isIndexSnapshot) return;
          const points = chart.getElementsAtEventForMode(event, 'nearest', { intersect: true }, false);
          if (!points || !points.length) return;
          const point = points[0];
          const clickedFeature = featureSnapshot[point.index]?.feature || '';
          const clickedRawFeature = featureSnapshot[point.index]?.rawFeature || '';
          const row = this.selectedRow();
          if (row) {
            this.openEntityModal(row, {
              date: this.state.date,
              featureType: clickedRawFeature,
              featureDisplayName: clickedFeature,
              tab: 'features',
            });
          }
        },
        interaction: isIndexSnapshot
          ? { mode: 'index', intersect: true }
          : { mode: 'nearest', intersect: true },
        scales: {
          x: {
            beginAtZero: true,
            stacked: isIndexSnapshot,
            ticks: { callback: (value) => `${value}%`, color: '#a6d7dd' },
            grid: { color: 'rgba(162,235,243,.08)' },
          },
          y: {
            stacked: isIndexSnapshot,
            ticks: { color: '#eaf5f5' },
            grid: { display: false },
          },
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (context) => {
                const item = featureSnapshot[context.dataIndex];
                if (isIndexSnapshot) {
                  if (context.datasetIndex === 0) {
                    return `${formatPercent(item?.negativePct, 1, 'N/A')} negative`;
                  }
                  return `${formatPercent(item?.nonNegativePct, 1, 'N/A')} positive/neutral`;
                }
                const negCount = Number(item?.negativeCount ?? 0);
                const totalCount = Number(item?.total ?? 0);
                return [
                  `${formatPercent(item?.negativePct, 1, 'N/A')} negative`,
                  `${formatInteger(negCount)} / ${formatInteger(totalCount)} negative URLs`,
                ];
              },
              footer: (items) => {
                if (!isIndexSnapshot || !items?.length) return '';
                const item = featureSnapshot[items[0].dataIndex];
                return `${formatPercent(item?.coveragePct, 1, 'N/A')} brands with feature`;
              },
            },
          },
        },
      },
    }, Chart);

    const featureCompositeView = this.state.featureCompositeView === 'area' ? 'area' : 'bar';
    const featureCompositeChartType = featureCompositeView === 'area' ? 'line' : 'bar';
    const featureCompositeDatasets = featureCompositeSeries.datasets.map((dataset, index) => {
      const color = FEATURE_COMPOSITE_COLORS[index % FEATURE_COMPOSITE_COLORS.length];
      return {
        label: featureTypeLabel(dataset.rawFeature || dataset.feature),
        data: dataset.values,
        rawFeature: dataset.rawFeature || '',
        stack: 'negative-feature-composite',
        backgroundColor: color,
        borderColor: color,
        borderRadius: featureCompositeView === 'bar' ? 6 : 0,
        borderWidth: featureCompositeView === 'area' ? 1.4 : 0,
        fill: featureCompositeView === 'area',
        pointRadius: 0,
        pointHoverRadius: 3,
        tension: featureCompositeView === 'area' ? 0.22 : 0,
      };
    });

    this.upsertChart('featureComposite', this.nodes.featureCompositeChart, {
      type: featureCompositeChartType,
      data: {
        labels: featureCompositeSeries.dates,
        datasets: featureCompositeDatasets,
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: { mode: 'index', intersect: false },
        onClick: (event, _elements, chart) => {
          const points = chart.getElementsAtEventForMode(event, 'nearest', { intersect: true }, false);
          if (!points || !points.length) return;
          const point = points[0];
          const clickedDate = featureCompositeSeries.dates[point.index];
          const clickedSeries = featureCompositeSeries.datasets[point.datasetIndex];
          if (!clickedDate || !clickedSeries?.rawFeature) return;
          openDateModalFromChart(clickedDate, {
            tab: 'features',
            featureType: clickedSeries.rawFeature,
            featureDisplayName: featureTypeLabel(clickedSeries.rawFeature),
          });
        },
        scales: {
          y: {
            beginAtZero: true,
            max: 100,
            stacked: true,
            ticks: { callback: (value) => `${value}%`, color: '#a6d7dd' },
            grid: { color: 'rgba(162,235,243,.08)' },
          },
          x: {
            stacked: true,
            ticks: { color: '#a6d7dd', maxRotation: 0, autoSkip: true, maxTicksLimit: 8 },
            grid: { color: 'rgba(162,235,243,.05)' },
          },
        },
        plugins: {
          legend: this.buildLegendTriStateOptions('featureComposite', { labels: { color: '#eaf5f5' } }),
          tooltip: {
            callbacks: {
              label: (context) => `${context.dataset.label}: ${formatPercentPoints(context.parsed.y, 1, 'N/A')}`,
            },
          },
        },
      },
    }, Chart);
    this.applyLegendTriStateStyles('featureComposite', this.charts.featureComposite, { update: false });

    const featurePresenceMetric = String(featurePresenceSeries?.metric || '').trim();
    const featurePresenceMetricConfig = FEATURE_PRESENCE_METRICS[featurePresenceMetric] || FEATURE_PRESENCE_METRICS.presence_rate;
    const isFeaturePresencePercentScale = featurePresenceMetricConfig.percentScale !== false;
    const isSerpSizeMetric = featurePresenceMetric === 'serp_size';
    const isSerpSizeStackedMetric = featurePresenceMetric === 'serp_size_stacked';
    const featurePresenceStacked = isSerpSizeStackedMetric;
    const featurePresenceDatasets = (featurePresenceSeries?.datasets || []).map((dataset, index) => {
      const rawFeature = String(dataset?.rawFeature || '').trim();
      const baseColor = FEATURE_TYPE_COLORS[rawFeature] || FEATURE_COMPOSITE_COLORS[index % FEATURE_COMPOSITE_COLORS.length];
      const color = isSerpSizeMetric
        ? 'rgba(147,228,255,.95)'
        : isSerpSizeStackedMetric
          ? baseColor
          : baseColor;
      const backgroundColor = isSerpSizeMetric
        ? 'rgba(147,228,255,.22)'
        : isSerpSizeStackedMetric
          ? withAlpha(baseColor, 0.5)
          : baseColor;
      return {
        label: featureTypeLabel(rawFeature || dataset?.feature || ''),
        data: Array.isArray(dataset?.values) ? dataset.values : [],
        details: Array.isArray(dataset?.details) ? dataset.details : [],
        rawFeature,
        borderColor: color,
        backgroundColor,
        pointBackgroundColor: color,
        pointBorderColor: color,
        borderWidth: isSerpSizeMetric ? 2.8 : isSerpSizeStackedMetric ? 1.4 : 2,
        fill: isSerpSizeMetric || isSerpSizeStackedMetric,
        stack: featurePresenceStacked ? 'feature-presence-slots' : undefined,
        pointRadius: 0,
        pointHoverRadius: (isSerpSizeMetric || isSerpSizeStackedMetric) ? 4 : 3,
        tension: (isSerpSizeMetric || isSerpSizeStackedMetric) ? 0.2 : 0.24,
        spanGaps: true,
      };
    });
    this.upsertChart('featurePresence', this.nodes.featurePresenceChart, {
      type: 'line',
      data: {
        labels: featurePresenceSeries?.dates || [],
        datasets: featurePresenceDatasets,
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: { mode: 'index', intersect: false },
        onClick: (event, _elements, chart) => {
          const points = chart.getElementsAtEventForMode(event, 'nearest', { intersect: false }, false);
          if (!points || !points.length) return;
          const point = points[0];
          const clickedDate = featurePresenceSeries?.dates?.[point.index] || chart?.data?.labels?.[point.index];
          const clickedDataset = featurePresenceSeries?.datasets?.[point.datasetIndex];
          if (!clickedDate) return;
          if (isSerpSizeMetric) {
            openDateModalFromChart(clickedDate, { tab: 'features' });
            return;
          }
          if (isSerpSizeStackedMetric) {
            const clickedFeature = String(clickedDataset?.rawFeature || '').trim();
            if (!clickedFeature) {
              openDateModalFromChart(clickedDate, { tab: 'features' });
              return;
            }
            openDateModalFromChart(clickedDate, {
              tab: 'features',
              featureType: clickedFeature,
              featureDisplayName: featureTypeLabel(clickedFeature),
            });
            return;
          }
          if (!clickedDataset?.rawFeature) return;
          openDateModalFromChart(clickedDate, {
            tab: 'features',
            featureType: clickedDataset.rawFeature,
            featureDisplayName: featureTypeLabel(clickedDataset.rawFeature),
          });
        },
        scales: {
          y: {
            beginAtZero: true,
            min: 0,
            stacked: featurePresenceStacked,
            max: isFeaturePresencePercentScale ? 100 : undefined,
            ticks: {
              stepSize: isFeaturePresencePercentScale ? 20 : undefined,
              callback: (value) => (isFeaturePresencePercentScale ? `${value}%` : formatInteger(value, '0')),
              color: '#a6d7dd',
            },
            grid: { color: 'rgba(162,235,243,.08)' },
          },
          x: {
            ticks: { color: '#a6d7dd', maxRotation: 0, autoSkip: true, maxTicksLimit: 8 },
            grid: { color: 'rgba(162,235,243,.05)' },
          },
        },
        plugins: {
          legend: this.buildLegendTriStateOptions('featurePresence', { labels: { color: '#eaf5f5' } }),
          tooltip: {
            enabled: false,
            external: (tooltipContext) => this.renderExternalChartTooltip(tooltipContext, { chartKey: 'feature-presence' }),
            callbacks: {
              label: (context) => {
                const details = context.dataset?.details?.[context.dataIndex] || null;
                const numerator = Number(details?.numerator) || 0;
                const denominator = Number(details?.denominator) || 0;
                if (!isFeaturePresencePercentScale) {
                  const slotCount = Number(context.parsed.y);
                  const avgSlotsPerActive = Number(details?.avgSlotsPerActive);
                  const pageOneSlots = Number(details?.pageOneSlots) || 0;
                  const lines = [
                    `${context.dataset.label}: ${formatInteger(slotCount, '0')} slots`,
                    `${formatInteger(numerator)}/${formatInteger(denominator)} ${featurePresenceMetricConfig.countLabel}`,
                  ];
                  if (isSerpSizeStackedMetric && pageOneSlots > 0) {
                    lines.push(`${((slotCount / pageOneSlots) * 100).toFixed(1)}% of total page-one slots`);
                  }
                  if (Number.isFinite(avgSlotsPerActive)) {
                    lines.push(`${avgSlotsPerActive.toFixed(2)} slots per active brand`);
                  }
                  return lines;
                }
                const rate = formatPercent(context.parsed.y, 1, 'N/A');
                return [
                  `${context.dataset.label}: ${rate}`,
                  `${formatInteger(numerator)}/${formatInteger(denominator)} ${featurePresenceMetricConfig.countLabel}`,
                ];
              },
              footer: (items) => {
                const first = Array.isArray(items) && items.length ? items[0] : null;
                if (!first) return '';
                const details = first.dataset?.details?.[first.dataIndex] || null;
                if (!details) return '';
                const total = Number(details.total) || 0;
                const negative = Number(details.negative) || 0;
                const controlled = Number(details.controlled) || 0;
                if (featurePresenceSeries.metric === 'serp_size') {
                  return `Page-one slots: ${formatInteger(details.pageOneSlots || 0)} · Active brands: ${formatInteger(details.activeCount || 0)}`;
                }
                if (featurePresenceSeries.metric === 'serp_size_stacked') {
                  return `Total page-one slots: ${formatInteger(details.pageOneSlots || 0)} · Active brands: ${formatInteger(details.activeCount || 0)}`;
                }
                if (featurePresenceSeries.metric === 'slot_share') {
                  return `Feature slots: ${formatInteger(total)} · Page-one slots: ${formatInteger(details.pageOneSlots || 0)}`;
                }
                if (featurePresenceSeries.metric === 'presence_rate') {
                  return `Brands with feature: ${formatInteger(details.presentCount || 0)} · Active brands: ${formatInteger(details.activeCount || 0)}`;
                }
                return `Negative: ${formatInteger(negative)} · Controlled: ${formatInteger(controlled)} · Total slots: ${formatInteger(total)}`;
              },
            },
          },
        },
      },
    }, Chart);
    this.applyLegendTriStateStyles('featurePresence', this.charts.featurePresence, { update: false });

    const isWindowLeaderboard = leaderboard?.mode === 'window';
    const leaderboardSeriesLabel = isWindowLeaderboard ? 'Adjusted composite signal' : 'Composite signal';
    const leaderboardMeta = Array.isArray(leaderboard?.meta) ? leaderboard.meta : [];
    const leaderboardWeights = normalizeSignalWeights(this.signalWeights, this.config.signalSettings?.weights);
    const ratioWithPct = (negative, total) => {
      const neg = Number(negative) || 0;
      const den = Number(total) || 0;
      if (den <= 0) return `${formatInteger(neg)}/${formatInteger(den)}`;
      return `${formatInteger(neg)}/${formatInteger(den)} (${((neg / den) * 100).toFixed(1)}%)`;
    };
    const componentPairs = (meta) => ([
      { label: 'News', negative: Number(meta?.newsNegative) || 0, total: Number(meta?.newsTotal) || 0 },
      { label: 'Organic', negative: Number(meta?.organicNegative) || 0, total: Number(meta?.organicTotal) || 0 },
      { label: 'Top stories', negative: Number(meta?.topStoriesNegative) || 0, total: Number(meta?.topStoriesTotal) || 0 },
      { label: 'AIO', negative: Number(meta?.aioNegative) || 0, total: Number(meta?.aioTotal) || 0 },
      { label: 'PAA', negative: Number(meta?.paaNegative) || 0, total: Number(meta?.paaTotal) || 0 },
      { label: 'Videos', negative: Number(meta?.videosNegative) || 0, total: Number(meta?.videosTotal) || 0 },
      { label: 'Perspectives', negative: Number(meta?.perspectivesNegative) || 0, total: Number(meta?.perspectivesTotal) || 0 },
    ]);
    const leaderboardBasisForMeta = (meta) => {
      const raw = String(meta?.listingBasis || '').trim().toLowerCase();
      if (raw === 'page_one' || raw === 'page-one') return 'page_one';
      return 'all';
    };
    const leaderboardBasisLabel = (basis) => (
      basis === 'page_one'
        ? 'Page-one only (organic + SERP features)'
        : 'All listings (news + page one)'
    );
    const leaderboardDatasets = LEADERBOARD_COMPONENTS.map((component) => ({
      label: component.label,
      data: leaderboard.values.map((value, index) => {
        const meta = leaderboardMeta[index];
        if (!meta) return 0;
        const basis = leaderboardBasisForMeta(meta);
        const totalScore = Math.max(0, Number(value) || 0);
        if (!Number.isFinite(totalScore) || totalScore <= 0) return 0;
        const weightedComponents = LEADERBOARD_COMPONENTS.reduce((acc, entry) => {
          const total = Number(meta?.[entry.totalKey]) || 0;
          const negative = Number(meta?.[entry.key]) || 0;
          const rate = total > 0 ? (negative / total) : 0;
          const weight = Number(leaderboardWeights?.[entry.weightKey]) || 0;
          const weighted = (basis === 'page_one' && entry.key === 'newsNegative') ? 0 : (rate * weight);
          acc[entry.key] = Math.max(0, weighted);
          return acc;
        }, {});
        const weightedTotal = Object.values(weightedComponents).reduce((sum, part) => sum + (Number(part) || 0), 0);
        if (weightedTotal <= 0) return 0;
        const componentWeighted = Number(weightedComponents[component.key]) || 0;
        if (componentWeighted <= 0) return 0;
        return totalScore * (componentWeighted / weightedTotal);
      }),
      borderRadius: 12,
      backgroundColor: component.color,
      stack: 'leaderboard-mix',
    }));

    this.upsertChart('leaderboard', this.nodes.leaderboardChart, {
      type: 'bar',
      data: {
        labels: leaderboard.labels,
        datasets: leaderboardDatasets,
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        indexAxis: 'y',
        onClick: (event, _elements, chart) => {
          const points = chart.getElementsAtEventForMode(event, 'nearest', { intersect: true }, false);
          if (!points || !points.length) return;
          const point = points[0];
          const label = leaderboard.labels[point.index];
          if (!label) return;
          const row = this.rows.find((item) => item.entity === label);
          if (row) this.openEntityModal(row, { date: this.state.date });
        },
        scales: {
          x: {
            beginAtZero: true,
            stacked: true,
            ticks: {
              color: '#a6d7dd',
              callback: (value) => (isWindowLeaderboard ? `${value}%` : `${value}`),
            },
            grid: { color: 'rgba(162,235,243,.08)' },
          },
          y: {
            stacked: true,
            ticks: { color: '#eaf5f5' },
            grid: { display: false },
          },
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (context) => {
                const value = Number(context.parsed.x);
                if (!Number.isFinite(value)) return '';
                if (isWindowLeaderboard) return `Score contribution · ${context.dataset.label}: ${value.toFixed(1)}%`;
                return `Score contribution · ${context.dataset.label}: ${value.toFixed(1)}`;
              },
              footer: (items) => {
                const first = Array.isArray(items) && items.length ? items[0] : null;
                const meta = first ? leaderboardMeta[first.dataIndex] : null;
                const totalValue = first ? Number(leaderboard.values[first.dataIndex] ?? 0) : 0;
                if (!meta) return '';
                const listingBasis = leaderboardBasisForMeta(meta);
                const listingLabel = listingBasis === 'page_one' ? 'Page-one listings' : 'Listings';
                const pairs = componentPairs(meta);
                const primaryPairs = pairs.slice(0, 2).map((entry) => `${entry.label} ${ratioWithPct(entry.negative, entry.total)}`);
                const featurePairs = pairs.slice(2)
                  .filter((entry) => entry.total > 0 || entry.negative > 0)
                  .map((entry) => `${entry.label} ${ratioWithPct(entry.negative, entry.total)}`);
                if (isWindowLeaderboard) {
                  return [
                    `Score (${leaderboardSeriesLabel.toLowerCase()}): ${totalValue.toFixed(1)}%`,
                    `${listingLabel}: ${ratioWithPct(meta.totalNegative, meta.totalListings)}`,
                    `Basis: ${leaderboardBasisLabel(listingBasis)}`,
                    `Days: ${formatInteger(meta.negativeDays)} negative / ${formatInteger(meta.activeDays)} active (${formatInteger(meta.zeroNegativeDays)} zero-negative)`,
                    `Avg visibility: ${Number(meta.avgDailyListings || 0).toFixed(1)} listings/day`,
                    `Raw composite: ${Number(meta.avgDailyScorePct || 0).toFixed(1)}% · Adjusted listing rate: ${Number(meta.adjustedRatePct || 0).toFixed(1)}%`,
                    `Core: ${primaryPairs.join(' · ')}`,
                    featurePairs.length
                      ? `SERP features: ${featurePairs.join(' · ')}`
                      : 'SERP features: none in this window',
                  ];
                }
                return [
                  `Score (${leaderboardSeriesLabel.toLowerCase()}): ${totalValue.toFixed(1)}`,
                  `Core: ${primaryPairs.join(' · ')}`,
                  featurePairs.length
                    ? `SERP features: ${featurePairs.join(' · ')}`
                    : 'SERP features: none on active date',
                ];
              },
            },
          },
        },
      },
    }, Chart);

    this.syncLeaderboardControls();
    this.nodes.newsPill.textContent = selectedEntity ? `${selectedEntity}` : `${visibleEntities.length} entities`;
    this.nodes.serpPill.textContent = selectedEntity ? `${selectedEntity}` : `${visibleEntities.length} entities`;
    this.nodes.featurePill.textContent = selectedEntity ? `${selectedEntity}` : `Index snapshot`;
    if (this.nodes.featureCompositePill) this.nodes.featureCompositePill.textContent = selectedEntity ? `${selectedEntity}` : 'Index composite';
    if (this.nodes.featurePresencePill) {
      const scopeLabel = selectedEntity ? selectedEntity : 'Index trends';
      this.nodes.featurePresencePill.textContent = `${scopeLabel} · ${featurePresenceMetricConfig.shortLabel}`;
    }
    if (this.nodes.leaderboardPill) {
      if (isWindowLeaderboard) {
        const minDaily = Math.max(0, Number(leaderboard?.minAvgDailyListings ?? this.state.leaderboardMinDailyListings) || 0);
        const listingBasis = String(leaderboard?.listingBasis || this.state.leaderboardListingBasis || '').trim().toLowerCase() === 'page_one'
          ? 'page_one'
          : 'all';
        const basisShort = listingBasis === 'page_one' ? 'page one' : 'all';
        this.nodes.leaderboardPill.textContent = `${leaderboard.labels.length} ranked · min ${minDaily}/day · ${basisShort}`;
      } else {
        this.nodes.leaderboardPill.textContent = `${leaderboard.labels.length} ranked`;
      }
    }
    if (this.nodes.leaderboardCaption) {
      if (isWindowLeaderboard) {
        const eligible = Number(leaderboard?.eligibleCandidates) || 0;
        const total = Number(leaderboard?.totalCandidates) || 0;
        const minDaily = Math.max(0, Number(leaderboard?.minAvgDailyListings ?? this.state.leaderboardMinDailyListings) || 0);
        const listingBasis = String(leaderboard?.listingBasis || this.state.leaderboardListingBasis || '').trim().toLowerCase() === 'page_one'
          ? 'page_one'
          : 'all';
        const basisLabel = listingBasis === 'page_one' ? 'page-one listings' : 'all listings';
        const summary = this.state.leaderboardDirection === 'best'
          ? `Lowest adjusted composite signal over ${this.state.days} days.`
          : `Highest adjusted composite signal over ${this.state.days} days.`;
        this.nodes.leaderboardCaption.textContent = `${summary} ${eligible}/${total} eligible at ${minDaily}+ ${basisLabel}/day.`;
      } else {
        this.nodes.leaderboardCaption.textContent = 'Highest combined crisis pressure on the active date.';
      }
    }
    this.nodes.newsCaption.textContent = selectedEntity
      ? `${selectedEntity} across ${this.state.days} days.`
      : `Visible ${this.config.label.toLowerCase()} across ${this.state.days} days.`;
    if (this.nodes.serpTitle) this.nodes.serpTitle.textContent = 'Organic Search Results';
    if (this.nodes.featureTitle) this.nodes.featureTitle.textContent = `Negative Page One Snapshot on ${this.state.date}`;
    if (this.nodes.featureCompositeTitle) this.nodes.featureCompositeTitle.textContent = 'Negative Page One SERP Composite';
    if (this.nodes.featurePresenceTitle) this.nodes.featurePresenceTitle.textContent = 'SERP Feature Trends';
    if (this.nodes.featurePresenceCaption) {
      this.nodes.featurePresenceCaption.textContent = `${featurePresenceMetricConfig.label}: ${featurePresenceSeries?.metricDescription || featurePresenceMetricConfig.description}.`;
    }
    this.syncFeaturePresenceMetricControl();
    this.updateHelpText(selectedEntity);
  }

  upsertChart(key, canvas, chartConfig, Chart) {
    if (!canvas) return;
    if (this.charts[key]) {
      const existing = this.charts[key];
      const existingType = String(existing?.config?.type || existing?.type || '').trim().toLowerCase();
      const nextType = String(chartConfig?.type || '').trim().toLowerCase();
      if (existingType && nextType && existingType !== nextType) {
        try {
          existing.destroy();
        } catch (_error) {
          // Ignore destroy failures and recreate fresh below.
        }
        delete this.charts[key];
      }
    }
    if (this.charts[key]) {
      const existing = this.charts[key];
      existing.data = chartConfig.data;
      existing.options = chartConfig.options;
      if (this.hasActiveLegendTriState(key)) {
        this.applyLegendTriStateStyles(key, existing, { update: false });
      }
      existing.update();
      return;
    }
    const context = canvas.getContext('2d');
    this.charts[key] = new Chart(context, chartConfig);
    if (this.hasActiveLegendTriState(key)) {
      this.applyLegendTriStateStyles(key, this.charts[key], { update: true });
    }
  }

  async refreshRowsAndVisuals({ updateCharts = true, replaceUrl = true } = {}) {
    const loadToken = ++this.loadToken;
    try {
      this.core = await this.store.ensureCore(this.state.days);
      if (!this.core.dates.length) {
        this.rows = [];
        this.renderSummary();
        this.renderSelected();
        this.renderTable();
        this.setFeedback('No entity data is available yet.');
        return;
      }

      if (!this.state.date || !this.core.dates.includes(this.state.date)) {
        this.state.date = this.core.dates[this.core.dates.length - 1];
      }
      this.updateLookbackButtons();
      this.updateDateSelect();

      const nextRows = await this.store.buildRows({
        days: this.state.days,
        date: this.state.date,
        query: this.state.query,
        signalWeights: this.signalWeights,
      });

      if (loadToken !== this.loadToken || this.destroyed) return;

      this.rows = this.sortRows(nextRows);
      if (this.state.selectedEntity && !this.rows.some((row) => row.entity === this.state.selectedEntity)) {
        this.state.selectedEntity = '';
      }
      if (!this.state.selectedEntity && this.rows.length === 1) {
        this.state.selectedEntity = this.rows[0].entity;
      }

      this.renderSummary();
      this.renderSelected();
      this.renderTable();
      this.updateCalibrationMetric();
      if (updateCharts) {
        await this.updateCharts();
      }
      this.updateUrl({ replace: true && replaceUrl });
      this.setFeedback(`Loaded ${this.rows.length} ${this.config.label.toLowerCase()} for ${this.state.date}.`);
    } catch (error) {
      console.error(error);
      this.setFeedback(error?.message || 'Failed to load the native entity tab.', 'error');
    }
  }

  async load({ replaceUrl = true } = {}) {
    this.setFeedback(`Loading ${this.config.label.toLowerCase()}…`);
    await this.refreshRowsAndVisuals({ updateCharts: true, replaceUrl });
    if (!this.supplementalLoadScheduled) {
      this.supplementalLoadScheduled = true;
      scheduleIdle(async () => {
        try {
          await this.store.ensureStockData();
          if (this.destroyed) return;
          await this.refreshRowsAndVisuals({ updateCharts: false, replaceUrl: false });
        } catch (error) {
          console.warn('Deferred stock load failed', error);
        }
      });
    }
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
    this.nodes.queryInput.value = this.state.query;
    await this.loadSharedPresets({ quiet: true });
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
    Object.values(this.charts).forEach((chart) => {
      try {
        chart.destroy();
      } catch (_error) {
        // Ignore Chart.js teardown failures.
      }
    });
    this.externalTooltipIds.forEach((id) => {
      const node = document.getElementById(id);
      if (node?.parentNode) node.parentNode.removeChild(node);
    });
    this.externalTooltipIds.clear();
    this.charts = {};
    this.host.innerHTML = '';
    this.host.hidden = true;
  }
}

export function isNativeEntityTab(tabId) {
  return isNativeEntityTabConfig(tabId);
}

export async function prefetchEntityTab({ tabId, shellConfig, getDirectUrl }) {
  const cacheKey = `${shellConfig.view || 'external'}:${tabId}`;
  if (prefetchRegistry.has(cacheKey)) return prefetchRegistry.get(cacheKey);

  const task = (async () => {
    const config = getEntityConfig(tabId, shellConfig);
    const store = getEntityStore(config);
    const currentUrl = getDirectUrl();
    const daysParam = Number(currentUrl.searchParams.get('days'));
    const days = config.lookbackOptions.includes(daysParam) ? daysParam : config.defaultDays;
    await Promise.all([
      ensureChartJs(),
      store.ensureCore(days),
    ]);
  })();

  prefetchRegistry.set(cacheKey, task);
  try {
    await task;
  } catch (error) {
    prefetchRegistry.delete(cacheKey);
    throw error;
  }
}

export async function mountEntityTab({ host, getDirectUrl, onHistoryChange, tabId, shellConfig }) {
  const controller = new EntityTabController({
    host,
    tabId,
    shellConfig,
    getDirectUrl,
    onHistoryChange,
  });
  await controller.init();
  return {
    show: () => controller.show(),
    hide: () => controller.hide(),
    destroy: () => controller.destroy(),
  };
}
