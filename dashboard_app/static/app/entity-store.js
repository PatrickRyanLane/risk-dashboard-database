import { clearSharedFetchCache, sharedFetch } from './shared-fetch.js';

const parsedJsonCache = new Map();
const latestDateCache = new Map();
const storeRegistry = new Map();

function normalizeUrl(url) {
  return new URL(url, window.location.origin).href;
}

function buildParsedCacheKey(url, init = {}) {
  const method = String(init.method || 'GET').toUpperCase();
  if (method !== 'GET') return '';
  return normalizeUrl(url);
}

async function fetchJson(url, init = {}) {
  const cacheKey = buildParsedCacheKey(url, init);
  if (cacheKey && parsedJsonCache.has(cacheKey)) {
    const cached = parsedJsonCache.get(cacheKey);
    return cached instanceof Promise ? await cached : cached;
  }

  const pending = (async () => {
    const response = await sharedFetch(url, { cache: 'default', credentials: 'same-origin', ...init });
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

  if (cacheKey) parsedJsonCache.set(cacheKey, pending);

  try {
    const data = await pending;
    if (cacheKey) parsedJsonCache.set(cacheKey, data);
    return data;
  } catch (error) {
    if (cacheKey) parsedJsonCache.delete(cacheKey);
    throw error;
  }
}

async function fetchLatestDatedJson({ key, buildUrl, maxDays = 7 }) {
  if (latestDateCache.has(key)) {
    const cachedDate = latestDateCache.get(key);
    const rows = await fetchJson(buildUrl(cachedDate));
    return { rows, date: cachedDate, daysBack: 0 };
  }

  const now = new Date();
  const todayUtc = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate()));
  let lastError = null;

  for (let daysBack = 0; daysBack < maxDays; daysBack += 1) {
    const checkDate = new Date(todayUtc);
    checkDate.setUTCDate(checkDate.getUTCDate() - daysBack);
    const dateStr = checkDate.toISOString().slice(0, 10);
    try {
      const rows = await fetchJson(buildUrl(dateStr));
      latestDateCache.set(key, dateStr);
      return { rows, date: dateStr, daysBack };
    } catch (error) {
      lastError = error;
    }
  }

  throw lastError || new Error(`No data found for ${key}`);
}

function collectDates(...collections) {
  const dates = new Set();
  collections.flat().forEach((row) => {
    const date = String(row?.date || '').trim();
    if (date) dates.add(date);
  });
  return Array.from(dates).sort((left, right) => left.localeCompare(right));
}

function pct01(value) {
  if (value == null || value === '') return null;
  const numeric = Number(String(value).replace('%', '').trim());
  if (!Number.isFinite(numeric)) return null;
  return numeric > 1 ? numeric / 100 : numeric;
}

function normEntityKey(value) {
  return String(value || '')
    .trim()
    .toLowerCase()
    .replace(/\s+/g, ' ')
    .replace(/[^a-z0-9]+/g, '');
}

function metricNumber(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : 0;
}

export const DEFAULT_SIGNAL_WEIGHTS = Object.freeze({
  newsNegative: 0.35,
  serpNegativeOrganic: 0.4,
  serpNegativeFeatures: 0.25,
  serpControl: 0.15,
});

function normalizeSignalWeights(weights = {}) {
  const toFinite = (value, fallback = 0) => {
    const numeric = Number(value);
    return Number.isFinite(numeric) ? numeric : fallback;
  };
  return {
    newsNegative: toFinite(weights.newsNegative, DEFAULT_SIGNAL_WEIGHTS.newsNegative),
    serpNegativeOrganic: toFinite(weights.serpNegativeOrganic, DEFAULT_SIGNAL_WEIGHTS.serpNegativeOrganic),
    serpNegativeFeatures: toFinite(weights.serpNegativeFeatures, DEFAULT_SIGNAL_WEIGHTS.serpNegativeFeatures),
    serpControl: toFinite(weights.serpControl, DEFAULT_SIGNAL_WEIGHTS.serpControl),
  };
}

export function computeCompositeSignal(row, signalWeights = DEFAULT_SIGNAL_WEIGHTS) {
  const weights = normalizeSignalWeights(signalWeights);
  const negNews = row.negNews ?? 0;
  const negSerp = row.negSerp ?? 0;
  const negFeatures = row.negFeatureAll ?? 0;
  const control = row.control ?? 0;
  return (
    (negNews * weights.newsNegative) +
    (negSerp * weights.serpNegativeOrganic) +
    (negFeatures * weights.serpNegativeFeatures) -
    (control * weights.serpControl)
  );
}

function computeRiskLabel(negPct, ctrlPct, threshold) {
  if (negPct == null || Number.isNaN(negPct)) return 'N/A';
  if (negPct > 0) return 'High';
  if (ctrlPct == null || Number.isNaN(ctrlPct)) return 'N/A';
  return ctrlPct < threshold ? 'Medium' : 'Low';
}

function formatFeatureName(value) {
  const raw = String(value || '').trim();
  if (!raw) return 'Unknown';
  const knownLabels = {
    organic: 'Organic',
    aio_citations: 'AIO citations',
    paa_items: 'PAA',
    videos_items: 'Videos',
    perspectives_items: 'Perspectives',
    top_stories_items: 'Top stories',
  };
  if (knownLabels[raw]) return knownLabels[raw];
  return raw
    .split('_')
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
}

const FEATURE_COMPOSITE_ORDER = [
  'organic',
  'aio_citations',
  'paa_items',
  'videos_items',
  'perspectives_items',
  'top_stories_items',
];

function parseCountsRows(config, rows) {
  return (Array.isArray(rows) ? rows : [])
    .map((row) => ({
      date: String(row.date || '').trim(),
      entity: String(row[config.primaryKey] || '').trim(),
      company: String(row.company || row[config.primaryKey] || '').trim(),
      positive: metricNumber(row.positive),
      neutral: metricNumber(row.neutral),
      negative: metricNumber(row.negative),
      total: metricNumber(row.total) || (
        metricNumber(row.positive) +
        metricNumber(row.neutral) +
        metricNumber(row.negative)
      ),
    }))
    .filter((row) => row.date && row.entity);
}

function parseSerpRows(config, rows) {
  return (Array.isArray(rows) ? rows : [])
    .map((row) => ({
      date: String(row.date || '').trim(),
      entity: String(row[config.primaryKey] || '').trim(),
      company: String(row.company || row[config.primaryKey] || '').trim(),
      total: metricNumber(row.total),
      negativeSerp: metricNumber(row.negative_serp ?? row.neg_serp),
      controlled: metricNumber(row.controlled ?? row.control),
    }))
    .filter((row) => row.date && row.entity);
}

function parseFeatureRows(rows) {
  return (Array.isArray(rows) ? rows : [])
    .map((row) => ({
      date: String(row.date || '').trim(),
      entity: String(row.entity_name || '').trim(),
      feature: String(row.feature_type || '').trim(),
      total: metricNumber(row.total_count || row.total),
      positive: metricNumber(row.positive_count || row.positive),
      neutral: metricNumber(row.neutral_count || row.neutral),
      negative: metricNumber(row.negative_count || row.negative),
      controlled: metricNumber(row.controlled_count || row.controlled),
    }))
    .filter((row) => row.date && row.feature);
}

function parseStockRows(rows) {
  const stockData = {};
  (Array.isArray(rows) ? rows : []).forEach((row) => {
    const company = String(row.company || '').trim();
    if (!company) return;
    stockData[company] = {
      company,
      ticker: String(row.ticker || '').trim(),
      openingPrice: Number.isFinite(Number(row.opening_price)) ? Number(row.opening_price) : null,
      dailyChange: Number.isFinite(Number(row.daily_change_pct ?? row.daily_change)) ? Number(row.daily_change_pct ?? row.daily_change) : null,
      sevenDayChange: Number.isFinite(Number(row.seven_day_change_pct)) ? Number(row.seven_day_change_pct) : null,
      priceHistory: Array.isArray(row.price_history)
        ? row.price_history.map(Number).filter(Number.isFinite)
        : String(row.price_history || '').split('|').map(Number).filter(Number.isFinite),
      dateHistory: Array.isArray(row.date_history)
        ? row.date_history
        : String(row.date_history || '').split('|').filter(Boolean),
    };
  });
  return stockData;
}

function parseRosterMeta(config, rows) {
  const meta = new Map();
  (Array.isArray(rows) ? rows : []).forEach((row) => {
    const company = String(row.company || row.Company || '').trim();
    const ceo = String(row.ceo || row.CEO || '').trim();
    const sector = String(row.sector || row.Sector || row.industry || row.Industry || '').trim();
    const ticker = String(row.ticker || row.Ticker || row.stock_ticker || row['Stock Ticker'] || '').trim();
    const favorite = ['true', '1', 'yes', 'y', 'x'].includes(String(
      row.favorite ||
      row.Favorite ||
      row.company_favorite ||
      row['Company Favorite'] ||
      row.ceo_favorite ||
      row['CEO Favorite'] ||
      ''
    ).trim().toLowerCase());

    if (config.tabId === 'brands' && company) {
      meta.set(company, {
        entity: company,
        company,
        sector,
        ticker,
        favorite,
      });
    }

    if (config.tabId === 'ceos' && ceo) {
      meta.set(ceo, {
        entity: ceo,
        company,
        sector,
        ticker,
        favorite,
      });
    }
  });
  return meta;
}

function ensureArrayRows(payload) {
  if (Array.isArray(payload)) return payload;
  if (Array.isArray(payload?.rows)) return payload.rows;
  return [];
}

class EntityStore {
  constructor(config) {
    this.config = config;
    this.coreByDays = new Map();
    this.topStoriesByDay = new Map();
    this.featureTotalsByDay = new Map();
    this.processedArticlesByDay = new Map();
    this.processedSerpsByDay = new Map();
    this.featureRowsByDate = new Map();
    this.featureRowsByEntity = new Map();
    this.featureControlRowsByEntity = new Map();
    this.rosterPromise = null;
    this.stockPromise = null;
    this.stockData = {};
  }

  clear() {
    this.coreByDays.clear();
    this.topStoriesByDay.clear();
    this.featureTotalsByDay.clear();
    this.processedArticlesByDay.clear();
    this.processedSerpsByDay.clear();
    this.featureRowsByDate.clear();
    this.featureRowsByEntity.clear();
    this.featureControlRowsByEntity.clear();
    this.rosterPromise = null;
    this.stockPromise = null;
    this.stockData = {};
    parsedJsonCache.clear();
    latestDateCache.clear();
    clearSharedFetchCache();
  }

  async ensureRoster() {
    if (this.rosterPromise) return this.rosterPromise;
    this.rosterPromise = (async () => {
      const rows = await fetchJson('/api/v1/roster');
      return parseRosterMeta(this.config, rows);
    })();
    return this.rosterPromise;
  }

  async ensureCore(days) {
    if (this.coreByDays.has(days)) {
      const cached = this.coreByDays.get(days);
      return cached instanceof Promise ? await cached : cached;
    }

    const pending = (async () => {
      const [countsPayload, serpsPayload, featuresPayload, featureControlsPayload, rosterMeta] = await Promise.all([
        fetchJson(`/api/v1/daily_counts?kind=${this.config.dailyCountsKind}&days=${days}`),
        fetchJson(`/api/v1/daily_counts?kind=${this.config.dailySerpKind}&days=${days}`),
        fetchJson(`/api/v1/serp_features?entity=${this.config.entityType}&days=${days}&mode=index`),
        fetchJson(`/api/v1/serp_feature_controls?entity=${this.config.entityType}&days=${days}&mode=index`),
        this.ensureRoster(),
      ]);

      const countsRows = parseCountsRows(this.config, countsPayload);
      const serpRows = parseSerpRows(this.config, serpsPayload);
      const featureIndexRows = parseFeatureRows(featuresPayload);
      const featureControlRows = parseFeatureRows(featureControlsPayload);
      const dates = collectDates(countsRows, serpRows, featureIndexRows);

      return {
        days,
        countsRows,
        serpRows,
        featureIndexRows,
        featureControlRows,
        rosterMeta,
        dates,
      };
    })();

    this.coreByDays.set(days, pending);
    try {
      const data = await pending;
      this.coreByDays.set(days, data);
      return data;
    } catch (error) {
      this.coreByDays.delete(days);
      throw error;
    }
  }

  async ensureStockData() {
    if (this.stockPromise) return this.stockPromise;
    this.stockPromise = (async () => {
      const payload = await fetchLatestDatedJson({
        key: 'entity_store_stock_data',
        buildUrl: (date) => `/api/v1/stock_data?date=${date}`,
        maxDays: 7,
      });
      this.stockData = parseStockRows(payload.rows);
      return this.stockData;
    })();
    return this.stockPromise;
  }

  async ensureTopStoriesForDate(days, date) {
    const cacheKey = `${days}:${date}`;
    if (this.topStoriesByDay.has(cacheKey)) {
      const cached = this.topStoriesByDay.get(cacheKey);
      return cached instanceof Promise ? await cached : cached;
    }

    const pending = (async () => {
      const rows = await fetchJson(`/api/v1/serp_features?entity=${this.config.entityType}&days=${days}&feature_type=top_stories_items&date=${date}`);
      const map = new Map();
      parseFeatureRows(rows).forEach((row) => {
        if (!row.entity) return;
        map.set(row.entity, row.total > 0 ? row.negative / row.total : null);
      });
      return map;
    })();

    this.topStoriesByDay.set(cacheKey, pending);
    try {
      const data = await pending;
      this.topStoriesByDay.set(cacheKey, data);
      return data;
    } catch (error) {
      this.topStoriesByDay.delete(cacheKey);
      throw error;
    }
  }

  async ensureFeatureTotalsForDate(days, date) {
    const cacheKey = `${days}:${date}`;
    if (this.featureTotalsByDay.has(cacheKey)) {
      const cached = this.featureTotalsByDay.get(cacheKey);
      return cached instanceof Promise ? await cached : cached;
    }

    const pending = (async () => {
      const rows = await this.ensureFeatureRowsForDate(days, date);
      const totalsByEntity = new Map();
      rows
        .filter((row) => row.date === date && row.entity)
        .forEach((row) => {
          const bucket = totalsByEntity.get(row.entity) || { total: 0, negative: 0 };
          bucket.total += row.total;
          bucket.negative += row.negative;
          totalsByEntity.set(row.entity, bucket);
        });
      return totalsByEntity;
    })();

    this.featureTotalsByDay.set(cacheKey, pending);
    try {
      const data = await pending;
      this.featureTotalsByDay.set(cacheKey, data);
      return data;
    } catch (error) {
      this.featureTotalsByDay.delete(cacheKey);
      throw error;
    }
  }

  async ensureProcessedArticles(date) {
    if (this.processedArticlesByDay.has(date)) {
      const cached = this.processedArticlesByDay.get(date);
      return cached instanceof Promise ? await cached : cached;
    }

    const pending = (async () => {
      const payload = await fetchJson(`/api/v1/processed_articles?date=${date}&entity=${this.config.processedEntity}&kind=table`);
      const rows = ensureArrayRows(payload);
      const map = new Map();
      rows.forEach((row) => {
        const entity = String(row[this.config.primaryKey] || '').trim();
        if (!entity) return;
        let negPct = pct01(row.neg_pct);
        if (negPct == null) {
          const negative = metricNumber(row.negative);
          const total = metricNumber(row.total) || (
            metricNumber(row.positive) +
            metricNumber(row.neutral) +
            negative
          );
          negPct = total ? negative / total : 0;
        }
        map.set(entity, {
          entity,
          company: String(row.company || entity).trim(),
          negPct,
        });
      });
      return map;
    })();

    this.processedArticlesByDay.set(date, pending);
    try {
      const data = await pending;
      this.processedArticlesByDay.set(date, data);
      return data;
    } catch (error) {
      this.processedArticlesByDay.delete(date);
      throw error;
    }
  }

  async ensureProcessedSerps(date) {
    if (this.processedSerpsByDay.has(date)) {
      const cached = this.processedSerpsByDay.get(date);
      return cached instanceof Promise ? await cached : cached;
    }

    const pending = (async () => {
      const payload = await fetchJson(`/api/v1/processed_serps?date=${date}&entity=${this.config.processedEntity}&kind=table`);
      const rows = ensureArrayRows(payload);
      const map = new Map();
      rows.forEach((row) => {
        const entity = String(row[this.config.primaryKey] || '').trim();
        if (!entity) return;
        map.set(entity, {
          entity,
          company: String(row.company || entity).trim(),
          total: metricNumber(row.total),
          negativeSerp: metricNumber(row.negative_serp),
          controlled: metricNumber(row.controlled),
        });
      });
      return map;
    })();

    this.processedSerpsByDay.set(date, pending);
    try {
      const data = await pending;
      this.processedSerpsByDay.set(date, data);
      return data;
    } catch (error) {
      this.processedSerpsByDay.delete(date);
      throw error;
    }
  }

  async ensureEntityFeatureRows(days, entity) {
    const cacheKey = `${days}:${entity}`;
    if (this.featureRowsByEntity.has(cacheKey)) {
      const cached = this.featureRowsByEntity.get(cacheKey);
      return cached instanceof Promise ? await cached : cached;
    }

    const pending = (async () => {
      const rows = await fetchJson(`/api/v1/serp_features?entity=${this.config.entityType}&days=${days}&entity_name=${encodeURIComponent(entity)}`);
      return parseFeatureRows(rows);
    })();

    this.featureRowsByEntity.set(cacheKey, pending);
    try {
      const data = await pending;
      this.featureRowsByEntity.set(cacheKey, data);
      return data;
    } catch (error) {
      this.featureRowsByEntity.delete(cacheKey);
      throw error;
    }
  }

  async ensureEntityFeatureControlRows(days, entity) {
    const cacheKey = `${days}:${entity}`;
    if (this.featureControlRowsByEntity.has(cacheKey)) {
      const cached = this.featureControlRowsByEntity.get(cacheKey);
      return cached instanceof Promise ? await cached : cached;
    }

    const pending = (async () => {
      const rows = await fetchJson(`/api/v1/serp_feature_controls?entity=${this.config.entityType}&days=${days}&entity_name=${encodeURIComponent(entity)}`);
      return parseFeatureRows(rows);
    })();

    this.featureControlRowsByEntity.set(cacheKey, pending);
    try {
      const data = await pending;
      this.featureControlRowsByEntity.set(cacheKey, data);
      return data;
    } catch (error) {
      this.featureControlRowsByEntity.delete(cacheKey);
      throw error;
    }
  }

  async buildRows({ days, date, query = '', signalWeights = null }) {
    const [core, articles, serpAgg, topStories, allFeatureTotals] = await Promise.all([
      this.ensureCore(days),
      this.ensureProcessedArticles(date),
      this.ensureProcessedSerps(date),
      this.ensureTopStoriesForDate(days, date),
      this.ensureFeatureTotalsForDate(days, date),
    ]);

    const search = String(query || '').trim().toLowerCase();
    const rows = [];

    for (const [entity, article] of articles.entries()) {
      const serp = serpAgg.get(entity);
      const metadata = core.rosterMeta.get(entity) || {};
      const company = this.config.tabId === 'ceos'
        ? (article.company || metadata.company || '')
        : entity;
      const stock = this.stockData[company] || null;
      const negSerp = serp && serp.total > 0 ? serp.negativeSerp / serp.total : null;
      const control = serp && serp.total > 0 ? serp.controlled / serp.total : null;
      const topStoriesPct = topStories.get(entity) ?? null;
      const featureTotals = allFeatureTotals.get(entity) || { total: 0, negative: 0 };
      const negFeatureAll = featureTotals.total > 0 ? (featureTotals.negative / featureTotals.total) : null;

      const row = {
        entity,
        company,
        sector: metadata.sector || '',
        favorite: !!metadata.favorite,
        ticker: metadata.ticker || stock?.ticker || '',
        negNews: article.negPct ?? 0,
        topStories: topStoriesPct,
        negFeatureAll,
        negFeatureAllCount: featureTotals.negative || 0,
        featureAllTotal: featureTotals.total || 0,
        negSerp,
        control,
        stock: stock?.dailyChange ?? null,
        risk: computeRiskLabel(negSerp, control, this.config.controlThreshold),
      };
      row.riskScore = computeCompositeSignal(row, signalWeights || this.config.signalSettings?.weights);

      const matches = !search
        || row.entity.toLowerCase().includes(search)
        || row.company.toLowerCase().includes(search)
        || row.sector.toLowerCase().includes(search);
      if (matches) rows.push(row);
    }

    return rows;
  }

  async buildNewsSeries(days, visibleEntities = [], selectedEntity = '') {
    const core = await this.ensureCore(days);
    if (!selectedEntity && Array.isArray(visibleEntities) && visibleEntities.length === 0) {
      return core.dates.map(() => null);
    }
    const focusSet = selectedEntity
      ? new Set([selectedEntity])
      : (visibleEntities.length ? new Set(visibleEntities) : null);
    const byDate = new Map();

    core.countsRows.forEach((row) => {
      if (focusSet && !focusSet.has(row.entity)) return;
      const bucket = byDate.get(row.date) || { negative: 0, total: 0 };
      bucket.negative += row.negative;
      bucket.total += row.total;
      byDate.set(row.date, bucket);
    });

    return core.dates.map((date) => {
      const bucket = byDate.get(date) || { negative: 0, total: 0 };
      return bucket.total ? (bucket.negative / bucket.total) * 100 : null;
    });
  }

  async buildSerpSeries(days, visibleEntities = [], selectedEntity = '') {
    const core = await this.ensureCore(days);
    if (!selectedEntity && Array.isArray(visibleEntities) && visibleEntities.length === 0) {
      return {
        dates: core.dates,
        negative: core.dates.map(() => null),
        control: core.dates.map(() => null),
      };
    }
    const focusSet = selectedEntity
      ? new Set([selectedEntity])
      : (visibleEntities.length ? new Set(visibleEntities) : null);
    const byDate = new Map();

    core.serpRows.forEach((row) => {
      if (focusSet && !focusSet.has(row.entity)) return;
      const bucket = byDate.get(row.date) || { negative: 0, controlled: 0, total: 0 };
      bucket.negative += row.negativeSerp;
      bucket.controlled += row.controlled;
      bucket.total += row.total;
      byDate.set(row.date, bucket);
    });

    return {
      dates: core.dates,
      negative: core.dates.map((date) => {
        const bucket = byDate.get(date) || { negative: 0, total: 0 };
        return bucket.total ? (bucket.negative / bucket.total) * 100 : null;
      }),
      control: core.dates.map((date) => {
        const bucket = byDate.get(date) || { controlled: 0, total: 0 };
        return bucket.total ? (bucket.controlled / bucket.total) * 100 : null;
      }),
    };
  }

  async ensureFeatureRowsForDate(days, date) {
    const cacheKey = `${days}:${date}`;
    if (this.featureRowsByDate.has(cacheKey)) {
      const cached = this.featureRowsByDate.get(cacheKey);
      return cached instanceof Promise ? await cached : cached;
    }

    const pending = (async () => {
      const rows = await fetchJson(`/api/v1/serp_features?entity=${this.config.entityType}&days=${days}&date=${date}`);
      return parseFeatureRows(rows);
    })();

    this.featureRowsByDate.set(cacheKey, pending);
    try {
      const data = await pending;
      this.featureRowsByDate.set(cacheKey, data);
      return data;
    } catch (error) {
      this.featureRowsByDate.delete(cacheKey);
      throw error;
    }
  }

  async buildFeatureSnapshot(days, date, selectedEntity = '', visibleEntities = []) {
    let featureRows;
    let controlRows;

    if (selectedEntity) {
      [featureRows, controlRows] = await Promise.all([
        this.ensureEntityFeatureRows(days, selectedEntity),
        this.ensureEntityFeatureControlRows(days, selectedEntity),
      ]);
      const controlLookup = new Map();
      controlRows
        .filter((row) => row.date === date)
        .forEach((row) => {
          controlLookup.set(row.feature, row.total > 0 ? row.controlled / row.total : null);
        });

      return featureRows
        .filter((row) => row.date === date)
        .map((row) => ({
          feature: formatFeatureName(row.feature),
          rawFeature: row.feature,
          negativePct: row.total > 0 ? (row.negative / row.total) * 100 : 0,
          negativeCount: row.negative,
          coveragePct: row.total > 0 ? 100 : 0,
          nonNegativePct: row.total > 0 ? ((row.total - row.negative) / row.total) * 100 : 0,
          controlPct: (controlLookup.get(row.feature) ?? 0) * 100,
          total: row.total,
        }))
        .sort((left, right) => right.negativePct - left.negativePct)
        .slice(0, 8);
    }

    const [core, featureRowsByDate] = await Promise.all([
      this.ensureCore(days),
      this.ensureFeatureRowsForDate(days, date),
    ]);
    const fallbackEntities = Array.from(new Set(
      core.countsRows
        .filter((row) => row.date === date && row.entity)
        .map((row) => normEntityKey(row.entity))
        .filter(Boolean),
    ));
    const focusKeys = Array.from(new Set(
      (Array.isArray(visibleEntities) && visibleEntities.length ? visibleEntities : fallbackEntities)
        .map((value) => normEntityKey(value))
        .filter(Boolean),
    ));
    if (!focusKeys.length) return [];
    const focusSet = new Set(focusKeys);
    const denominator = focusSet.size;
    const byFeature = new Map();

    featureRowsByDate
      .filter((row) => row.date === date && row.entity)
      .forEach((row) => {
        const entityKey = normEntityKey(row.entity);
        if (!focusSet.has(entityKey)) return;
        const bucket = byFeature.get(row.feature) || { withFeatureEntities: new Set(), negativeEntities: new Set() };
        if (row.total > 0) {
          bucket.withFeatureEntities.add(entityKey);
          if (row.negative > 0) bucket.negativeEntities.add(entityKey);
        }
        byFeature.set(row.feature, bucket);
      });

    return Array.from(byFeature.entries())
      .map(([rawFeature, bucket]) => {
        const withFeature = bucket.withFeatureEntities.size;
        const negative = bucket.negativeEntities.size;
        const coveragePct = denominator > 0 ? (withFeature / denominator) * 100 : 0;
        const negativePct = denominator > 0 ? (negative / denominator) * 100 : 0;
        const nonNegativePct = Math.max(0, coveragePct - negativePct);
        return {
          feature: formatFeatureName(rawFeature),
          rawFeature,
          negativePct,
          negativeCount: negative,
          coveragePct,
          nonNegativePct,
          controlPct: 0,
          total: withFeature,
        };
      })
      .filter((row) => row.coveragePct > 0)
      .sort((left, right) => right.coveragePct - left.coveragePct)
      .slice(0, 8);
  }

  async buildFeatureCompositeSeries(days, selectedEntity = '') {
    const core = await this.ensureCore(days);
    const dates = core.dates.slice();
    const dateSet = new Set(dates);
    let sourceRows = core.featureIndexRows;
    if (selectedEntity) {
      sourceRows = await this.ensureEntityFeatureRows(days, selectedEntity);
    }

    const allowedItemFeatures = new Set(FEATURE_COMPOSITE_ORDER.filter((feature) => feature !== 'organic'));
    const byFeature = new Map();
    const totalSlotsByDate = new Map();

    const addSlotCounts = (feature, date, negativeSlots, totalSlots) => {
      if (!dateSet.has(date)) return;
      const total = metricNumber(totalSlots);
      const negative = metricNumber(negativeSlots);
      if (total <= 0 && negative <= 0) return;
      const byDate = byFeature.get(feature) || new Map();
      const bucket = byDate.get(date) || { negative: 0, total: 0 };
      bucket.negative += negative;
      bucket.total += total;
      byDate.set(date, bucket);
      byFeature.set(feature, byDate);
      totalSlotsByDate.set(date, (totalSlotsByDate.get(date) || 0) + total);
    };

    sourceRows.forEach((row) => {
      if (!row?.feature || !row?.date || !dateSet.has(row.date)) return;
      const feature = String(row.feature).trim();
      if (!feature || !allowedItemFeatures.has(feature)) return;
      addSlotCounts(feature, row.date, row.negative, row.total);
    });

    core.serpRows.forEach((row) => {
      if (!row?.date || !dateSet.has(row.date)) return;
      if (selectedEntity && row.entity !== selectedEntity) return;
      addSlotCounts('organic', row.date, row.negativeSerp, row.total);
    });

    const knownOrder = FEATURE_COMPOSITE_ORDER.filter((feature) => byFeature.has(feature));
    const unknownOrder = Array.from(byFeature.keys())
      .filter((feature) => !FEATURE_COMPOSITE_ORDER.includes(feature))
      .sort((left, right) => left.localeCompare(right));
    const featureOrder = [...knownOrder, ...unknownOrder];

    return {
      dates,
      datasets: featureOrder.map((feature) => {
        const byDate = byFeature.get(feature) || new Map();
        return {
          rawFeature: feature,
          feature: formatFeatureName(feature),
          values: dates.map((date) => {
            const dayTotalSlots = totalSlotsByDate.get(date) || 0;
            if (dayTotalSlots <= 0) return null;
            const bucket = byDate.get(date) || { negative: 0 };
            return (bucket.negative / dayTotalSlots) * 100;
          }),
        };
      }),
    };
  }

  async buildLeaderboard(days, date, rows) {
    const snapshot = rows
      .slice()
      .sort((left, right) => right.riskScore - left.riskScore)
      .slice(0, 10);
    return {
      date,
      labels: snapshot.map((row) => row.entity),
      values: snapshot.map((row) => Number((row.riskScore * 100).toFixed(2))),
    };
  }
}

export function getEntityStore(config) {
  const key = `${config.view}:${config.tabId}`;
  if (!storeRegistry.has(key)) {
    storeRegistry.set(key, new EntityStore(config));
  }
  return storeRegistry.get(key);
}

export function clearAllEntityStores() {
  storeRegistry.forEach((store) => store.clear());
}
