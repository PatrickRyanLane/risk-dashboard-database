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

function normalizeFeatureType(value) {
  const raw = String(value || '')
    .trim()
    .toLowerCase()
    .replace(/[\s-]+/g, '_');
  if (!raw) return '';
  const aliases = {
    topstories: 'top_stories_items',
    top_stories: 'top_stories_items',
    top_stories_item: 'top_stories_items',
    aio: 'aio_citations',
    aio_citation: 'aio_citations',
    paa: 'paa_items',
    paa_item: 'paa_items',
    videos: 'videos_items',
    videos_item: 'videos_items',
    perspectives: 'perspectives_items',
    perspectives_item: 'perspectives_items',
    organic_results: 'organic',
    organic_serp: 'organic',
  };
  return aliases[raw] || raw;
}

export const DEFAULT_SIGNAL_WEIGHTS = Object.freeze({
  newsNegative: 0.24,
  organicNegative: 0.24,
  topStoriesNegative: 0.16,
  aioCitationsNegative: 0.12,
  paaNegative: 0.1,
  videosNegative: 0.07,
  perspectivesNegative: 0.07,
  serpControl: 0.1,
});

function normalizeSignalWeights(weights = {}) {
  const toFinite = (value, fallback = 0) => {
    const numeric = Number(value);
    return Number.isFinite(numeric) ? numeric : fallback;
  };
  const legacyFeatures = toFinite(
    weights.serpNegativeFeatures,
    (
      DEFAULT_SIGNAL_WEIGHTS.topStoriesNegative
      + DEFAULT_SIGNAL_WEIGHTS.aioCitationsNegative
      + DEFAULT_SIGNAL_WEIGHTS.paaNegative
      + DEFAULT_SIGNAL_WEIGHTS.videosNegative
      + DEFAULT_SIGNAL_WEIGHTS.perspectivesNegative
    ),
  );
  const featureFallbackSum = (
    DEFAULT_SIGNAL_WEIGHTS.topStoriesNegative
    + DEFAULT_SIGNAL_WEIGHTS.aioCitationsNegative
    + DEFAULT_SIGNAL_WEIGHTS.paaNegative
    + DEFAULT_SIGNAL_WEIGHTS.videosNegative
    + DEFAULT_SIGNAL_WEIGHTS.perspectivesNegative
  ) || 1;
  const featureScale = legacyFeatures / featureFallbackSum;
  return {
    newsNegative: toFinite(weights.newsNegative, DEFAULT_SIGNAL_WEIGHTS.newsNegative),
    organicNegative: toFinite(
      weights.organicNegative,
      toFinite(weights.serpNegativeOrganic, DEFAULT_SIGNAL_WEIGHTS.organicNegative),
    ),
    topStoriesNegative: toFinite(
      weights.topStoriesNegative,
      DEFAULT_SIGNAL_WEIGHTS.topStoriesNegative * featureScale,
    ),
    aioCitationsNegative: toFinite(
      weights.aioCitationsNegative,
      DEFAULT_SIGNAL_WEIGHTS.aioCitationsNegative * featureScale,
    ),
    paaNegative: toFinite(
      weights.paaNegative,
      DEFAULT_SIGNAL_WEIGHTS.paaNegative * featureScale,
    ),
    videosNegative: toFinite(
      weights.videosNegative,
      DEFAULT_SIGNAL_WEIGHTS.videosNegative * featureScale,
    ),
    perspectivesNegative: toFinite(
      weights.perspectivesNegative,
      DEFAULT_SIGNAL_WEIGHTS.perspectivesNegative * featureScale,
    ),
    serpControl: toFinite(weights.serpControl, DEFAULT_SIGNAL_WEIGHTS.serpControl),
  };
}

export function computeCompositeSignal(row, signalWeights = DEFAULT_SIGNAL_WEIGHTS) {
  const weights = normalizeSignalWeights(signalWeights);
  const negNews = row.negNews ?? 0;
  const negSerp = row.negSerp ?? 0;
  const negTopStories = row.negTopStories ?? row.topStories ?? 0;
  const negAio = row.negAio ?? 0;
  const negPaa = row.negPaa ?? 0;
  const negVideos = row.negVideos ?? 0;
  const negPerspectives = row.negPerspectives ?? 0;
  const control = row.control ?? 0;
  return (
    (negNews * weights.newsNegative) +
    (negSerp * weights.organicNegative) +
    (negTopStories * weights.topStoriesNegative) +
    (negAio * weights.aioCitationsNegative) +
    (negPaa * weights.paaNegative) +
    (negVideos * weights.videosNegative) +
    (negPerspectives * weights.perspectivesNegative) -
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
      feature: normalizeFeatureType(row.feature_type),
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
    this.featureRowsByWindow = new Map();
    this.featureControlRowsByWindow = new Map();
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
    this.featureRowsByWindow.clear();
    this.featureControlRowsByWindow.clear();
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
          const bucket = totalsByEntity.get(row.entity) || { total: 0, negative: 0, byFeature: {} };
          bucket.total += row.total;
          bucket.negative += row.negative;
          const featureKey = String(row.feature || '').trim();
          if (featureKey) {
            const featureBucket = bucket.byFeature[featureKey] || { total: 0, negative: 0 };
            featureBucket.total += row.total;
            featureBucket.negative += row.negative;
            bucket.byFeature[featureKey] = featureBucket;
          }
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
        const negative = metricNumber(row.negative);
        let total = metricNumber(row.total) || (
          metricNumber(row.positive) +
          metricNumber(row.neutral) +
          negative
        );
        if (negPct == null) {
          negPct = total ? negative / total : 0;
        }
        if (!total && negPct != null) {
          // Keep denominator meaningful if only percentage was provided.
          total = 1;
        }
        map.set(entity, {
          entity,
          company: String(row.company || entity).trim(),
          negPct,
          negative,
          total,
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
      const featureTotals = allFeatureTotals.get(entity) || { total: 0, negative: 0, byFeature: {} };
      const featureByType = featureTotals.byFeature || {};
      const featurePct = (featureKey) => {
        const bucket = featureByType[featureKey];
        const total = metricNumber(bucket?.total);
        if (!total) return 0;
        return metricNumber(bucket?.negative) / total;
      };
      const featureCounts = (featureKey) => {
        const bucket = featureByType[featureKey];
        return {
          negative: metricNumber(bucket?.negative),
          total: metricNumber(bucket?.total),
        };
      };
      const articleNegative = metricNumber(article.negative);
      const articleTotal = metricNumber(article.total);
      const serpNegative = metricNumber(serp?.negativeSerp);
      const serpTotal = metricNumber(serp?.total);
      const topStoriesCounts = featureCounts('top_stories_items');
      const aioCounts = featureCounts('aio_citations');
      const paaCounts = featureCounts('paa_items');
      const videosCounts = featureCounts('videos_items');
      const perspectivesCounts = featureCounts('perspectives_items');
      const pageOneNegative = featureTotals.negative + serpNegative;
      const pageOneTotal = featureTotals.total + serpTotal;
      const negFeatureAll = pageOneTotal > 0 ? (pageOneNegative / pageOneTotal) : null;
      const negSerpFeatures = featureTotals.total > 0 ? (featureTotals.negative / featureTotals.total) : null;

      const row = {
        entity,
        company,
        sector: metadata.sector || '',
        favorite: !!metadata.favorite,
        ticker: metadata.ticker || stock?.ticker || '',
        negNews: article.negPct ?? 0,
        negNewsCount: articleNegative,
        newsTotal: articleTotal,
        negTopStories: topStoriesPct ?? featurePct('top_stories_items'),
        negAio: featurePct('aio_citations'),
        negPaa: featurePct('paa_items'),
        negVideos: featurePct('videos_items'),
        negPerspectives: featurePct('perspectives_items'),
        topStories: topStoriesPct ?? featurePct('top_stories_items'),
        negSerpFeatures,
        negFeatureAll,
        negFeatureAllCount: pageOneNegative || 0,
        featureAllTotal: pageOneTotal || 0,
        negSerpFeatureCount: featureTotals.negative || 0,
        serpFeatureTotal: featureTotals.total || 0,
        negTopStoriesCount: topStoriesCounts.negative,
        topStoriesTotal: topStoriesCounts.total,
        negAioCount: aioCounts.negative,
        aioTotal: aioCounts.total,
        negPaaCount: paaCounts.negative,
        paaTotal: paaCounts.total,
        negVideosCount: videosCounts.negative,
        videosTotal: videosCounts.total,
        negPerspectivesCount: perspectivesCounts.negative,
        perspectivesTotal: perspectivesCounts.total,
        negSerp,
        negSerpCount: serpNegative,
        serpTotal,
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

  async ensureFeatureRowsForWindow(days) {
    if (this.featureRowsByWindow.has(days)) {
      const cached = this.featureRowsByWindow.get(days);
      return cached instanceof Promise ? await cached : cached;
    }

    const pending = (async () => {
      const rows = await fetchJson(`/api/v1/serp_features?entity=${this.config.entityType}&days=${days}`);
      return parseFeatureRows(rows);
    })();

    this.featureRowsByWindow.set(days, pending);
    try {
      const data = await pending;
      this.featureRowsByWindow.set(days, data);
      return data;
    } catch (error) {
      this.featureRowsByWindow.delete(days);
      throw error;
    }
  }

  async ensureFeatureControlRowsForWindow(days) {
    if (this.featureControlRowsByWindow.has(days)) {
      const cached = this.featureControlRowsByWindow.get(days);
      return cached instanceof Promise ? await cached : cached;
    }

    const pending = (async () => {
      const rows = await fetchJson(`/api/v1/serp_feature_controls?entity=${this.config.entityType}&days=${days}`);
      return parseFeatureRows(rows);
    })();

    this.featureControlRowsByWindow.set(days, pending);
    try {
      const data = await pending;
      this.featureControlRowsByWindow.set(days, data);
      return data;
    } catch (error) {
      this.featureControlRowsByWindow.delete(days);
      throw error;
    }
  }

  async buildFeatureSnapshot(days, date, selectedEntity = '', visibleEntities = []) {
    let featureRows;
    let controlRows;

    if (selectedEntity) {
      const [core, serpAgg] = await Promise.all([
        this.ensureCore(days),
        this.ensureProcessedSerps(date),
      ]);
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

      const featureItems = featureRows
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
        }));

      const organicRows = core.serpRows.filter((row) => row.entity === selectedEntity && row.date === date);
      const organicTotal = organicRows.reduce((sum, row) => sum + metricNumber(row.total), 0);
      const organicNegative = organicRows.reduce((sum, row) => sum + metricNumber(row.negativeSerp), 0);
      const organicControlled = organicRows.reduce((sum, row) => sum + metricNumber(row.controlled), 0);
      if (organicTotal > 0) {
        featureItems.push({
          feature: formatFeatureName('organic'),
          rawFeature: 'organic',
          negativePct: (organicNegative / organicTotal) * 100,
          negativeCount: organicNegative,
          coveragePct: 100,
          nonNegativePct: ((organicTotal - organicNegative) / organicTotal) * 100,
          controlPct: (organicControlled / organicTotal) * 100,
          total: organicTotal,
        });
      }

      return featureItems
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

    core.serpRows
      .filter((row) => row.date === date && row.entity)
      .forEach((row) => {
        const entityKey = normEntityKey(row.entity);
        if (!focusSet.has(entityKey)) return;
        const bucket = byFeature.get('organic') || { withFeatureEntities: new Set(), negativeEntities: new Set() };
        if (metricNumber(row.total) > 0) {
          bucket.withFeatureEntities.add(entityKey);
          if (metricNumber(row.negativeSerp) > 0) bucket.negativeEntities.add(entityKey);
        }
        byFeature.set('organic', bucket);
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

  async buildFeaturePresenceSeries(days, visibleEntities = [], selectedEntity = '', metric = 'slot_share') {
    const [core, featureRowsForWindow, controlRowsForWindow] = await Promise.all([
      this.ensureCore(days),
      selectedEntity ? Promise.resolve([]) : this.ensureFeatureRowsForWindow(days),
      selectedEntity ? Promise.resolve([]) : this.ensureFeatureControlRowsForWindow(days),
    ]);
    const dates = core.dates.slice();
    const dateSet = new Set(dates);
    const normalizedMetric = (() => {
      const raw = String(metric || '').trim().toLowerCase();
      if (
        raw === 'presence_rate'
        || raw === 'slot_share'
        || raw === 'negative_share'
        || raw === 'controlled_share'
        || raw === 'serp_size'
        || raw === 'serp_size_stacked'
      ) {
        return raw;
      }
      return 'slot_share';
    })();
    const metricLabelByKey = {
      presence_rate: 'Presence rate',
      slot_share: 'Slot share',
      negative_share: 'Negative share',
      controlled_share: 'Controlled share',
      serp_size: 'Page-One Real Estate',
      serp_size_stacked: 'Page-One Real Estate (stacked)',
    };
    const metricDescriptionByKey = {
      presence_rate: 'Brands with feature / active brands',
      slot_share: 'Feature slots / all page-one slots',
      negative_share: 'Negative slots / feature slots',
      controlled_share: 'Controlled slots / feature slots',
      serp_size: 'Total page-one slots per day (organic + SERP features)',
      serp_size_stacked: 'Total page-one slots per day, split by feature (organic + SERP features)',
    };

    const focusSet = selectedEntity
      ? new Set([String(selectedEntity).trim()])
      : new Set(
        (Array.isArray(visibleEntities) ? visibleEntities : [])
          .map((value) => String(value || '').trim())
          .filter(Boolean),
      );
    const includeEntity = (entityName) => {
      if (!entityName) return false;
      if (selectedEntity) return entityName === selectedEntity;
      return focusSet.size === 0 || focusSet.has(entityName);
    };

    const [sourceFeatureRows, sourceControlRows] = selectedEntity
      ? await Promise.all([
        this.ensureEntityFeatureRows(days, selectedEntity),
        this.ensureEntityFeatureControlRows(days, selectedEntity),
      ])
      : [featureRowsForWindow, controlRowsForWindow];

    const featureByDate = new Map();
    const entityPageOneTotalsByDate = new Map();
    const totalPageOneSlotsByDate = new Map();

    const ensureFeatureDateBucket = (feature, date) => {
      const normalizedFeature = normalizeFeatureType(feature);
      if (!normalizedFeature || !dateSet.has(date)) return null;
      const byDate = featureByDate.get(normalizedFeature) || new Map();
      const bucket = byDate.get(date) || {
        total: 0,
        negative: 0,
        controlled: 0,
        entitiesWithFeature: new Set(),
      };
      byDate.set(date, bucket);
      featureByDate.set(normalizedFeature, byDate);
      return bucket;
    };

    const addEntityPageOneSlots = (date, entity, slots) => {
      if (!dateSet.has(date) || !entity) return;
      const numericSlots = metricNumber(slots);
      if (numericSlots <= 0) return;
      const byEntity = entityPageOneTotalsByDate.get(date) || new Map();
      byEntity.set(entity, (byEntity.get(entity) || 0) + numericSlots);
      entityPageOneTotalsByDate.set(date, byEntity);
      totalPageOneSlotsByDate.set(date, (totalPageOneSlotsByDate.get(date) || 0) + numericSlots);
    };

    sourceFeatureRows.forEach((row) => {
      const date = String(row?.date || '').trim();
      const entity = String(row?.entity || '').trim();
      const feature = normalizeFeatureType(row?.feature);
      if (!date || !entity || !feature || !includeEntity(entity)) return;
      if (feature === 'organic') return;
      const total = metricNumber(row.total);
      const negative = metricNumber(row.negative);
      const bucket = ensureFeatureDateBucket(feature, date);
      if (!bucket) return;
      bucket.total += total;
      bucket.negative += negative;
      if (total > 0) bucket.entitiesWithFeature.add(entity);
      addEntityPageOneSlots(date, entity, total);
    });

    sourceControlRows.forEach((row) => {
      const date = String(row?.date || '').trim();
      const entity = String(row?.entity || '').trim();
      const feature = normalizeFeatureType(row?.feature);
      if (!date || !entity || !feature || !includeEntity(entity)) return;
      if (feature === 'organic') return;
      const bucket = ensureFeatureDateBucket(feature, date);
      if (!bucket) return;
      bucket.controlled += metricNumber(row.controlled);
    });

    core.serpRows.forEach((row) => {
      const date = String(row?.date || '').trim();
      const entity = String(row?.entity || '').trim();
      if (!date || !entity || !includeEntity(entity) || !dateSet.has(date)) return;
      const total = metricNumber(row.total);
      const negative = metricNumber(row.negativeSerp);
      const controlled = metricNumber(row.controlled);
      const bucket = ensureFeatureDateBucket('organic', date);
      if (!bucket) return;
      bucket.total += total;
      bucket.negative += negative;
      bucket.controlled += controlled;
      if (total > 0) bucket.entitiesWithFeature.add(entity);
      addEntityPageOneSlots(date, entity, total);
    });

    const activeEntitiesByDate = new Map();
    dates.forEach((date) => {
      const byEntity = entityPageOneTotalsByDate.get(date);
      if (!byEntity) {
        activeEntitiesByDate.set(date, new Set());
        return;
      }
      const activeSet = new Set();
      byEntity.forEach((totalSlots, entity) => {
        if (metricNumber(totalSlots) > 0) activeSet.add(entity);
      });
      activeEntitiesByDate.set(date, activeSet);
    });

    const knownOrder = FEATURE_COMPOSITE_ORDER.filter((feature) => featureByDate.has(feature));
    const unknownOrder = Array.from(featureByDate.keys())
      .filter((feature) => !FEATURE_COMPOSITE_ORDER.includes(feature))
      .sort((left, right) => left.localeCompare(right));
    const featureOrder = [...knownOrder, ...unknownOrder];

    if (normalizedMetric === 'serp_size') {
      return {
        metric: normalizedMetric,
        metricLabel: metricLabelByKey[normalizedMetric] || metricLabelByKey.slot_share,
        metricDescription: metricDescriptionByKey[normalizedMetric] || metricDescriptionByKey.slot_share,
        dates,
        datasets: [
          {
            rawFeature: 'all_page_one_slots',
            feature: 'All page-one slots',
            values: dates.map((date) => {
              const pageOneSlots = metricNumber(totalPageOneSlotsByDate.get(date));
              return pageOneSlots > 0 ? pageOneSlots : null;
            }),
            details: dates.map((date) => {
              const pageOneSlots = metricNumber(totalPageOneSlotsByDate.get(date));
              const activeCount = (activeEntitiesByDate.get(date) || new Set()).size;
              return {
                numerator: pageOneSlots,
                denominator: activeCount,
                total: pageOneSlots,
                negative: 0,
                controlled: 0,
                presentCount: activeCount,
                activeCount,
                pageOneSlots,
                avgSlotsPerActive: activeCount > 0 ? pageOneSlots / activeCount : null,
              };
            }),
          },
        ],
      };
    }

    if (normalizedMetric === 'serp_size_stacked') {
      return {
        metric: normalizedMetric,
        metricLabel: metricLabelByKey[normalizedMetric] || metricLabelByKey.slot_share,
        metricDescription: metricDescriptionByKey[normalizedMetric] || metricDescriptionByKey.slot_share,
        dates,
        datasets: featureOrder.map((feature) => {
          const byDate = featureByDate.get(feature) || new Map();
          return {
            rawFeature: feature,
            feature: formatFeatureName(feature),
            values: dates.map((date) => {
              const pageOneSlots = metricNumber(totalPageOneSlotsByDate.get(date));
              if (pageOneSlots <= 0) return null;
              const bucket = byDate.get(date) || { total: 0 };
              return metricNumber(bucket.total);
            }),
            details: dates.map((date) => {
              const pageOneSlots = metricNumber(totalPageOneSlotsByDate.get(date));
              const bucket = byDate.get(date) || {
                total: 0,
                negative: 0,
                controlled: 0,
                entitiesWithFeature: new Set(),
              };
              const total = metricNumber(bucket.total);
              const activeCount = (activeEntitiesByDate.get(date) || new Set()).size;
              return {
                numerator: total,
                denominator: pageOneSlots,
                total,
                negative: metricNumber(bucket.negative),
                controlled: metricNumber(bucket.controlled),
                presentCount: bucket.entitiesWithFeature.size,
                activeCount,
                pageOneSlots,
                avgSlotsPerActive: activeCount > 0 ? total / activeCount : null,
              };
            }),
          };
        }),
      };
    }

    return {
      metric: normalizedMetric,
      metricLabel: metricLabelByKey[normalizedMetric] || metricLabelByKey.slot_share,
      metricDescription: metricDescriptionByKey[normalizedMetric] || metricDescriptionByKey.slot_share,
      dates,
      datasets: featureOrder.map((feature) => {
        const byDate = featureByDate.get(feature) || new Map();
        return {
          rawFeature: feature,
          feature: formatFeatureName(feature),
          values: dates.map((date) => {
            const bucket = byDate.get(date) || {
              total: 0,
              negative: 0,
              controlled: 0,
              entitiesWithFeature: new Set(),
            };
            const total = metricNumber(bucket.total);
            const negative = metricNumber(bucket.negative);
            const controlled = metricNumber(bucket.controlled);
            const activeCount = (activeEntitiesByDate.get(date) || new Set()).size;
            const presentCount = bucket.entitiesWithFeature.size;
            const pageOneSlots = metricNumber(totalPageOneSlotsByDate.get(date));
            if (normalizedMetric === 'presence_rate') {
              return activeCount > 0 ? (presentCount / activeCount) * 100 : null;
            }
            if (normalizedMetric === 'slot_share') {
              return pageOneSlots > 0 ? (total / pageOneSlots) * 100 : null;
            }
            if (normalizedMetric === 'negative_share') {
              return total > 0 ? (negative / total) * 100 : null;
            }
            if (normalizedMetric === 'controlled_share') {
              return total > 0 ? (controlled / total) * 100 : null;
            }
            return null;
          }),
          details: dates.map((date) => {
            const bucket = byDate.get(date) || {
              total: 0,
              negative: 0,
              controlled: 0,
              entitiesWithFeature: new Set(),
            };
            const total = metricNumber(bucket.total);
            const negative = metricNumber(bucket.negative);
            const controlled = metricNumber(bucket.controlled);
            const activeCount = (activeEntitiesByDate.get(date) || new Set()).size;
            const presentCount = bucket.entitiesWithFeature.size;
            const pageOneSlots = metricNumber(totalPageOneSlotsByDate.get(date));
            const numerator = normalizedMetric === 'presence_rate'
              ? presentCount
              : normalizedMetric === 'slot_share'
                ? total
                : normalizedMetric === 'negative_share'
                  ? negative
                  : controlled;
            const denominator = normalizedMetric === 'presence_rate'
              ? activeCount
              : normalizedMetric === 'slot_share'
                ? pageOneSlots
                : total;
            return {
              numerator,
              denominator,
              total,
              negative,
              controlled,
              presentCount,
              activeCount,
              pageOneSlots,
            };
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
      mode: 'date',
      date,
      labels: snapshot.map((row) => row.entity),
      values: snapshot.map((row) => Number((row.riskScore * 100).toFixed(2))),
      meta: snapshot.map((row) => ({
        entity: row.entity,
        newsNegative: metricNumber(row.negNewsCount),
        newsTotal: metricNumber(row.newsTotal),
        organicNegative: metricNumber(row.negSerpCount),
        organicTotal: metricNumber(row.serpTotal),
        topStoriesNegative: metricNumber(row.negTopStoriesCount),
        topStoriesTotal: metricNumber(row.topStoriesTotal),
        aioNegative: metricNumber(row.negAioCount),
        aioTotal: metricNumber(row.aioTotal),
        paaNegative: metricNumber(row.negPaaCount),
        paaTotal: metricNumber(row.paaTotal),
        videosNegative: metricNumber(row.negVideosCount),
        videosTotal: metricNumber(row.videosTotal),
        perspectivesNegative: metricNumber(row.negPerspectivesCount),
        perspectivesTotal: metricNumber(row.perspectivesTotal),
        featureNegative: metricNumber(row.negSerpFeatureCount),
        featureTotal: metricNumber(row.serpFeatureTotal),
        pageOneNegative: metricNumber(row.negFeatureAllCount),
        pageOneTotal: metricNumber(row.featureAllTotal),
      })),
    };
  }

  async buildWindowLeaderboard({
    days,
    visibleEntities = [],
    signalWeights = null,
    direction = 'worst',
    limit = 10,
    restrictToVisible = false,
    minAvgDailyListings = 0,
    listingBasis = 'all',
  } = {}) {
    const [core, featureRowsForWindow] = await Promise.all([
      this.ensureCore(days),
      this.ensureFeatureRowsForWindow(days),
    ]);
    const normalizedDirection = String(direction || '').trim().toLowerCase() === 'best' ? 'best' : 'worst';
    const normalizedListingBasis = String(listingBasis || '').trim().toLowerCase() === 'page_one' ? 'page_one' : 'all';
    const usePageOneBasis = normalizedListingBasis === 'page_one';
    const cap = Math.max(1, Number(limit) || 10);
    const minDailyThreshold = Math.max(0, Number(minAvgDailyListings) || 0);
    const weights = normalizeSignalWeights(signalWeights || this.config.signalSettings?.weights);
    const componentKeys = [
      'newsNegative',
      'organicNegative',
      'topStoriesNegative',
      'aioNegative',
      'paaNegative',
      'videosNegative',
      'perspectivesNegative',
    ];
    const focusSet = new Set(
      (Array.isArray(visibleEntities) ? visibleEntities : [])
        .map((value) => String(value || '').trim())
        .filter(Boolean),
    );
    const useVisibleScope = !!restrictToVisible;
    if (useVisibleScope && focusSet.size === 0) {
      return {
        mode: 'window',
        direction: normalizedDirection,
        listingBasis: normalizedListingBasis,
        windowStart: core.dates[0] || '',
        windowEnd: core.dates[core.dates.length - 1] || '',
        minAvgDailyListings: minDailyThreshold,
        totalCandidates: 0,
        eligibleCandidates: 0,
        labels: [],
        values: [],
        meta: [],
      };
    }
    const includeEntity = (entityName) => !useVisibleScope || focusSet.has(entityName);

    const byEntity = new Map();
    const ensureEntityBucket = (entityName) => {
      if (byEntity.has(entityName)) return byEntity.get(entityName);
      const bucket = {
        entity: entityName,
        newsNegative: 0,
        newsTotal: 0,
        organicNegative: 0,
        organicTotal: 0,
        topStoriesNegative: 0,
        topStoriesTotal: 0,
        aioNegative: 0,
        aioTotal: 0,
        paaNegative: 0,
        paaTotal: 0,
        videosNegative: 0,
        videosTotal: 0,
        perspectivesNegative: 0,
        perspectivesTotal: 0,
        featureNegative: 0,
        featureTotal: 0,
        controlled: 0,
        byDate: new Map(),
      };
      byEntity.set(entityName, bucket);
      return bucket;
    };
    const ensureDateBucket = (entityBucket, date) => {
      if (entityBucket.byDate.has(date)) return entityBucket.byDate.get(date);
      const bucket = {
        newsNegative: 0,
        newsTotal: 0,
        organicNegative: 0,
        organicTotal: 0,
        topStoriesNegative: 0,
        topStoriesTotal: 0,
        aioNegative: 0,
        aioTotal: 0,
        paaNegative: 0,
        paaTotal: 0,
        videosNegative: 0,
        videosTotal: 0,
        perspectivesNegative: 0,
        perspectivesTotal: 0,
        featureNegative: 0,
        featureTotal: 0,
        controlled: 0,
      };
      entityBucket.byDate.set(date, bucket);
      return bucket;
    };

    core.countsRows.forEach((row) => {
      const entity = String(row?.entity || '').trim();
      const date = String(row?.date || '').trim();
      if (!entity || !date || !includeEntity(entity)) return;
      const bucket = ensureEntityBucket(entity);
      const day = ensureDateBucket(bucket, date);
      bucket.newsNegative += metricNumber(row.negative);
      bucket.newsTotal += metricNumber(row.total);
      day.newsNegative += metricNumber(row.negative);
      day.newsTotal += metricNumber(row.total);
    });

    core.serpRows.forEach((row) => {
      const entity = String(row?.entity || '').trim();
      const date = String(row?.date || '').trim();
      if (!entity || !date || !includeEntity(entity)) return;
      const bucket = ensureEntityBucket(entity);
      const day = ensureDateBucket(bucket, date);
      bucket.organicNegative += metricNumber(row.negativeSerp);
      bucket.organicTotal += metricNumber(row.total);
      bucket.controlled += metricNumber(row.controlled);
      day.organicNegative += metricNumber(row.negativeSerp);
      day.organicTotal += metricNumber(row.total);
      day.controlled += metricNumber(row.controlled);
    });

    featureRowsForWindow.forEach((row) => {
      const entity = String(row?.entity || '').trim();
      const date = String(row?.date || '').trim();
      const rawFeature = String(row?.feature || '').trim();
      if (!entity || !date || !rawFeature || !includeEntity(entity)) return;
      if (rawFeature === 'organic') return;
      const bucket = ensureEntityBucket(entity);
      const day = ensureDateBucket(bucket, date);
      const negative = metricNumber(row.negative);
      const total = metricNumber(row.total);
      bucket.featureNegative += negative;
      bucket.featureTotal += total;
      day.featureNegative += negative;
      day.featureTotal += total;
      if (rawFeature === 'top_stories_items') {
        bucket.topStoriesNegative += negative;
        bucket.topStoriesTotal += total;
        day.topStoriesNegative += negative;
        day.topStoriesTotal += total;
      } else if (rawFeature === 'aio_citations') {
        bucket.aioNegative += negative;
        bucket.aioTotal += total;
        day.aioNegative += negative;
        day.aioTotal += total;
      } else if (rawFeature === 'paa_items') {
        bucket.paaNegative += negative;
        bucket.paaTotal += total;
        day.paaNegative += negative;
        day.paaTotal += total;
      } else if (rawFeature === 'videos_items') {
        bucket.videosNegative += negative;
        bucket.videosTotal += total;
        day.videosNegative += negative;
        day.videosTotal += total;
      } else if (rawFeature === 'perspectives_items') {
        bucket.perspectivesNegative += negative;
        bucket.perspectivesTotal += total;
        day.perspectivesNegative += negative;
        day.perspectivesTotal += total;
      }
    });

    const rows = Array.from(byEntity.values()).map((bucket) => {
      const pageOneNegative = bucket.organicNegative + bucket.featureNegative;
      const pageOneTotal = bucket.organicTotal + bucket.featureTotal;
      const allNegative = bucket.newsNegative + pageOneNegative;
      const allListings = bucket.newsTotal + pageOneTotal;
      const totalNegative = usePageOneBasis ? pageOneNegative : allNegative;
      const totalListings = usePageOneBasis ? pageOneTotal : allListings;
      const rawRate = totalListings > 0 ? totalNegative / totalListings : null;

      let negativeDays = 0;
      let zeroNegativeDays = 0;
      let activeDays = 0;
      let scoreSum = 0;
      let scoreDays = 0;
      const componentScoreSums = {
        newsNegative: 0,
        organicNegative: 0,
        topStoriesNegative: 0,
        aioNegative: 0,
        paaNegative: 0,
        videosNegative: 0,
        perspectivesNegative: 0,
      };
      core.dates.forEach((date) => {
        const day = bucket.byDate.get(date);
        if (!day) return;
        const dayPageOneNegative = day.organicNegative + day.featureNegative;
        const dayPageOneTotal = day.organicTotal + day.featureTotal;
        const dayAllNegative = day.newsNegative + dayPageOneNegative;
        const dayAllTotal = day.newsTotal + dayPageOneTotal;
        const dayNegative = usePageOneBasis ? dayPageOneNegative : dayAllNegative;
        const dayTotal = usePageOneBasis ? dayPageOneTotal : dayAllTotal;
        if (dayTotal <= 0) return;
        activeDays += 1;
        if (dayNegative > 0) negativeDays += 1;
        else zeroNegativeDays += 1;
        const dayNewsRate = day.newsTotal > 0 ? (day.newsNegative / day.newsTotal) : 0;
        const dayOrganicRate = day.organicTotal > 0 ? (day.organicNegative / day.organicTotal) : 0;
        const dayTopStoriesRate = day.topStoriesTotal > 0 ? (day.topStoriesNegative / day.topStoriesTotal) : 0;
        const dayAioRate = day.aioTotal > 0 ? (day.aioNegative / day.aioTotal) : 0;
        const dayPaaRate = day.paaTotal > 0 ? (day.paaNegative / day.paaTotal) : 0;
        const dayVideosRate = day.videosTotal > 0 ? (day.videosNegative / day.videosTotal) : 0;
        const dayPerspectivesRate = day.perspectivesTotal > 0 ? (day.perspectivesNegative / day.perspectivesTotal) : 0;
        const dayComponentScores = {
          newsNegative: (usePageOneBasis ? 0 : dayNewsRate) * weights.newsNegative,
          organicNegative: dayOrganicRate * weights.organicNegative,
          topStoriesNegative: dayTopStoriesRate * weights.topStoriesNegative,
          aioNegative: dayAioRate * weights.aioCitationsNegative,
          paaNegative: dayPaaRate * weights.paaNegative,
          videosNegative: dayVideosRate * weights.videosNegative,
          perspectivesNegative: dayPerspectivesRate * weights.perspectivesNegative,
        };
        const dayScore = computeCompositeSignal({
          negNews: usePageOneBasis ? 0 : dayNewsRate,
          negSerp: dayOrganicRate,
          negTopStories: dayTopStoriesRate,
          negAio: dayAioRate,
          negPaa: dayPaaRate,
          negVideos: dayVideosRate,
          negPerspectives: dayPerspectivesRate,
          negFeatureAll: dayPageOneTotal > 0 ? (dayPageOneNegative / dayPageOneTotal) : 0,
          control: day.organicTotal > 0 ? (day.controlled / day.organicTotal) : 0,
        }, weights);
        if (Number.isFinite(dayScore)) {
          scoreSum += dayScore;
          scoreDays += 1;
          componentKeys.forEach((key) => {
            componentScoreSums[key] += Number(dayComponentScores[key] || 0);
          });
        }
      });
      const avgComponentScores = componentKeys.reduce((acc, key) => {
        acc[key] = scoreDays > 0 ? (componentScoreSums[key] / scoreDays) : 0;
        return acc;
      }, {});

      return {
        entity: bucket.entity,
        totalNegative,
        totalListings,
        allNegative,
        allListings,
        featureNegative: bucket.featureNegative,
        featureTotal: bucket.featureTotal,
        topStoriesNegative: bucket.topStoriesNegative,
        topStoriesTotal: bucket.topStoriesTotal,
        aioNegative: bucket.aioNegative,
        aioTotal: bucket.aioTotal,
        paaNegative: bucket.paaNegative,
        paaTotal: bucket.paaTotal,
        videosNegative: bucket.videosNegative,
        videosTotal: bucket.videosTotal,
        perspectivesNegative: bucket.perspectivesNegative,
        perspectivesTotal: bucket.perspectivesTotal,
        pageOneNegative,
        pageOneTotal,
        newsNegative: bucket.newsNegative,
        newsTotal: bucket.newsTotal,
        organicNegative: bucket.organicNegative,
        organicTotal: bucket.organicTotal,
        rawRate,
        scoreSum,
        scoreDays,
        avgDailyScore: scoreDays > 0 ? (scoreSum / scoreDays) : null,
        avgComponentScores,
        negativeDays,
        zeroNegativeDays,
        activeDays,
      };
    }).filter((row) => row.totalListings > 0);

    const globalNegative = rows.reduce((sum, row) => sum + row.totalNegative, 0);
    const globalTotal = rows.reduce((sum, row) => sum + row.totalListings, 0);
    const priorRate = globalTotal > 0 ? (globalNegative / globalTotal) : 0;
    const globalScoreWeight = rows.reduce((sum, row) => sum + Math.max(0, Number(row.scoreDays) || 0), 0);
    const globalScore = globalScoreWeight > 0
      ? rows.reduce((sum, row) => sum + ((Number(row.avgDailyScore) || 0) * Math.max(0, Number(row.scoreDays) || 0)), 0) / globalScoreWeight
      : 0;
    const globalComponentScores = componentKeys.reduce((acc, key) => {
      acc[key] = globalScoreWeight > 0
        ? rows.reduce(
          (sum, row) => sum + ((Number(row.avgComponentScores?.[key]) || 0) * Math.max(0, Number(row.scoreDays) || 0)),
          0,
        ) / globalScoreWeight
        : 0;
      return acc;
    }, {});
    const smoothingK = Math.max(30, core.dates.length * 2);
    const scorePriorDays = Math.max(7, Math.round(core.dates.length / 3));
    rows.forEach((row) => {
      row.adjustedRate = (row.totalNegative + (smoothingK * priorRate)) / (row.totalListings + smoothingK);
      const rowScoreDays = Math.max(0, Number(row.scoreDays) || 0);
      const adjustedScoreDenominator = rowScoreDays + scorePriorDays;
      row.adjustedScore = (
        (rowScoreDays * (Number(row.avgDailyScore) || 0))
        + (scorePriorDays * globalScore)
      ) / adjustedScoreDenominator;
      row.adjustedComponentScores = componentKeys.reduce((acc, key) => {
        acc[key] = (
          (rowScoreDays * (Number(row.avgComponentScores?.[key]) || 0))
          + (scorePriorDays * (Number(globalComponentScores[key]) || 0))
        ) / adjustedScoreDenominator;
        return acc;
      }, {});
      row.avgDailyListings = row.activeDays > 0 ? (row.totalListings / row.activeDays) : row.totalListings;
      row.metricPct = row.adjustedScore * 100;
    });

    const totalCandidates = rows.length;
    const eligibleRows = rows.filter((row) => row.avgDailyListings >= minDailyThreshold);

    eligibleRows.sort((left, right) => {
      if (normalizedDirection === 'best') {
        return (
          (left.metricPct - right.metricPct)
          || (right.avgDailyListings - left.avgDailyListings)
          || (right.totalListings - left.totalListings)
          || left.entity.localeCompare(right.entity)
        );
      }
      return (
        (right.metricPct - left.metricPct)
        || (right.negativeDays - left.negativeDays)
        || (right.totalListings - left.totalListings)
        || left.entity.localeCompare(right.entity)
      );
    });

    const snapshot = eligibleRows.slice(0, cap);
    return {
      mode: 'window',
      direction: normalizedDirection,
      listingBasis: normalizedListingBasis,
      windowStart: core.dates[0] || '',
      windowEnd: core.dates[core.dates.length - 1] || '',
      minAvgDailyListings: minDailyThreshold,
      totalCandidates,
      eligibleCandidates: eligibleRows.length,
      labels: snapshot.map((row) => row.entity),
      values: snapshot.map((row) => Number((row.metricPct).toFixed(2))),
      meta: snapshot.map((row) => ({
        entity: row.entity,
        listingBasis: normalizedListingBasis,
        totalNegative: row.totalNegative,
        totalListings: row.totalListings,
        allNegative: row.allNegative,
        allListings: row.allListings,
        rawRatePct: Number(((row.rawRate || 0) * 100).toFixed(2)),
        adjustedRatePct: Number(((row.adjustedRate || 0) * 100).toFixed(2)),
        avgDailyScorePct: Number(((row.avgDailyScore || 0) * 100).toFixed(2)),
        adjustedScorePct: Number((row.metricPct).toFixed(2)),
        adjustedComponentScores: row.adjustedComponentScores,
        negativeDays: row.negativeDays,
        zeroNegativeDays: row.zeroNegativeDays,
        activeDays: row.activeDays,
        avgDailyListings: Number((row.avgDailyListings || 0).toFixed(2)),
        newsNegative: row.newsNegative,
        newsTotal: row.newsTotal,
        organicNegative: row.organicNegative,
        organicTotal: row.organicTotal,
        topStoriesNegative: row.topStoriesNegative,
        topStoriesTotal: row.topStoriesTotal,
        aioNegative: row.aioNegative,
        aioTotal: row.aioTotal,
        paaNegative: row.paaNegative,
        paaTotal: row.paaTotal,
        videosNegative: row.videosNegative,
        videosTotal: row.videosTotal,
        perspectivesNegative: row.perspectivesNegative,
        perspectivesTotal: row.perspectivesTotal,
        featureNegative: row.featureNegative,
        featureTotal: row.featureTotal,
        pageOneNegative: row.pageOneNegative,
        pageOneTotal: row.pageOneTotal,
      })),
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
